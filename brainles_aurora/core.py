# basics
import os
import time
from typing import Dict

import monai
import nibabel as nib
import numpy as np
# dl
import torch
from monai.data import list_data_collate
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import BasicUNet
from monai.transforms import (Compose, EnsureChannelFirstd, Lambdad,
                              LoadImageD, RandGaussianNoised,
                              ScaleIntensityRangePercentilesd, ToTensord)
from path import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from brainles_aurora.aux import turbo_path
from brainles_aurora.download import download_model_weights
from brainles_aurora.constants import ModalityMode, ModelSelection


LIB_ABSPATH: str = os.path.dirname(os.path.abspath(__file__))

MODEL_WEIGHTS_DIR = os.path.join(LIB_ABSPATH, "model_weights")
if not os.path.exists(MODEL_WEIGHTS_DIR):
    download_model_weights(target_folder=LIB_ABSPATH)


def infer(
    segmentation_file,
    t1_file=None,
    t1c_file=None,
    t2_file=None,
    fla_file=None,
    whole_network_outputs_file=None,
    metastasis_network_outputs_file=None,
    cuda_devices="0",
    tta=True,
    # faster for single interference (on RTX 3090)
    sliding_window_batch_size=1,
    workers=0,
    threshold=0.5,
    sliding_window_overlap=0.5,
    crop_size=(192, 192, 32),
    model_selection=ModelSelection.BEST,
        verbosity=True):
    # configure logger
    # transform inputs to paths
    if t1_file is not None:
        t1_file = turbo_path(t1_file)

    if t1c_file is not None:
        t1c_file = turbo_path(t1c_file)

    if t2_file is not None:
        t2_file = turbo_path(t2_file)

    if fla_file is not None:
        fla_file = turbo_path(fla_file)
    # mode
    mode: ModalityMode = _determine_mode(t1_file, t1c_file, t2_file, fla_file)

    # cuda device settings
    device = _configure_device(cuda_devices)
    # dataloader
    data_loader = _get_dloader(
        mode=mode,
        t1_file=t1_file,
        t1c_file=t1c_file,
        t2_file=t2_file,
        fla_file=fla_file,
        workers=workers,
    )
    # load model
    model = _get_model(
        mode=mode,
        model_selection=model_selection,
        device=device
    )
    # create inferrer

    inferer = SlidingWindowInferer(
        roi_size=crop_size,  # = patch_size
        sw_batch_size=sliding_window_batch_size,
        sw_device=device,
        device=device,
        overlap=sliding_window_overlap,
        mode="gaussian",
        padding_mode="replicate",
    )

    # evaluate
    with torch.no_grad():
        model.eval()
        # loop through batches
        for data in tqdm(data_loader, 0):
            inputs = data["images"]

            outputs = inferer(inputs, model)
            if tta:
                outputs = _apply_test_time_augmentations(
                    data, inferer, model
                )

            # generate segmentation nifti
            try:
                reference_file = data["t1c"][0]
            except:
                try:
                    reference_file = data["fla"][0]
                except:
                    reference_file = data["t1"][0]
                else:
                    FileNotFoundError("no reference file found!")

            _create_nifti_seg(
                threshold=threshold,
                reference_file=reference_file,
                onehot_model_outputs_CHWD=outputs,
                output_file=segmentation_file,
                whole_network_output_file=whole_network_outputs_file,
                enhancing_network_output_file=metastasis_network_outputs_file,
            )


def _configure_device(cuda_devices: str) -> torch.device:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # clean memory
    torch.cuda.empty_cache()
    return device


def _determine_mode(
    t1_file: Path | None,
    t1c_file: Path | None,
    t2_file: Path | None,
    fla_file: Path | None,
) -> ModalityMode:
    def _present(file: Path | None) -> bool:
        return False if file == None else os.path.exists(file)

    t1, t1c, t2, fla = map(_present, [t1_file, t1c_file, t2_file, fla_file])

    print(
        f"t1: {t1} t1c: {t1c} t2: {t2} flair: {fla}"
    )

    possible_modes = [
        (t1, t1c, t2, fla, ModalityMode.T1_T1C_T2_FLA),
        (t1, t1c, not t2, fla,  ModalityMode.T1C_T1_FLA),
        (t1, t1c, not t2, not fla, ModalityMode.T1C_T1),
        (not t1, t1c, not t2, fla, ModalityMode.T1C_FLA),
        (not t1, t1c, not t2, not fla, ModalityMode.T1C_O),
        (not t1, not t1c, not t2, fla, ModalityMode.FLA_O),
        (t1, not t1c, not t2, not fla, ModalityMode.T1_O),
    ]

    mode = None
    for pm in possible_modes:
        if all(pm[:-1]):
            mode = pm[-1]
            break

    if not mode:
        raise NotImplementedError(
            "No model implemented for this combination of files")

    print("mode:", mode)
    return mode


def _get_dloader(
    mode: ModalityMode,
    t1_file: Path | None,
    t1c_file: Path | None,
    t2_file: Path | None,
    fla_file: Path | None,
    workers: int,
):
    # init transforms
    transforms = [
        LoadImageD(keys=["images"]),
        Lambdad(["images"], np.nan_to_num),
        ScaleIntensityRangePercentilesd(
            keys="images",
            lower=0.5,
            upper=99.5,
            b_min=0,
            b_max=1,
            clip=True,
            relative=False,
            channel_wise=True,
        ),
        ToTensord(keys=["images"]),
    ]
    if mode in [ModalityMode.T1_O, ModalityMode.T1C_O, ModalityMode.FLA_O]:
        transforms.insert(1, EnsureChannelFirstd(keys="images"))
    inference_transforms = Compose(transforms)

    # mode dict mapping
    mode_dict_mapping = {
        ModalityMode.T1_T1C_T2_FLA: {
            "t1": t1_file,
            "t1c": t1c_file,
            "t2": t2_file,
            "fla": fla_file,
            "images": [t1_file, t1c_file, t2_file, fla_file]
        },
        ModalityMode.T1C_T1_FLA: {
            "t1": t1_file,
            "t1c": t1c_file,
            "fla": fla_file,
            "images": [t1_file, t1c_file, fla_file]
        },
        ModalityMode.T1C_T1: {
            "t1": t1_file,
            "t1c": t1c_file,
            "images": [t1_file, t1c_file]
        },
        ModalityMode.T1C_FLA: {
            "t1c": t1c_file,
            "fla": fla_file,
            "images": [t1c_file, fla_file]
        },
        ModalityMode.T1C_O: {
            "t1c": t1c_file,
            "images": [t1c_file]
        },
        ModalityMode.FLA_O: {
            "fla": fla_file,
            "images": [fla_file]
        },
        ModalityMode.T1_O: {
            "t1": t1_file,
            "images": [t1_file]
        },
    }

    data = mode_dict_mapping.get(mode, None)
    if not data:
        # TODO: probably not needed, method should only be called if a valid mode is selected
        raise NotImplementedError(
            "No model implemented for this combination of files")

    #  instantiate dataset and dataloader
    infererence_ds = monai.data.Dataset(
        data=[data],
        transform=inference_transforms,
    )

    data_loader = DataLoader(
        infererence_ds,
        batch_size=1,
        num_workers=workers,
        collate_fn=list_data_collate,
        shuffle=False,
    )
    return data_loader


def _get_model(mode: ModalityMode, model_selection: ModelSelection, device: torch.device):
    mode_inchannels_mapping = {
        ModalityMode.T1_T1C_T2_FLA: 4,
        ModalityMode.T1C_T1_FLA: 3,
        ModalityMode.T1C_T1: 2,
        ModalityMode.T1C_FLA: 2,
        ModalityMode.T1C_O: 1,
        ModalityMode.FLA_O: 1,
        ModalityMode.T1_O: 1
    }

    # init model
    model = BasicUNet(
        spatial_dims=3,
        in_channels=mode_inchannels_mapping[mode],
        out_channels=2,
        features=(32, 32, 64, 128, 256, 32),
        dropout=0.1,
        act="mish",
    )

    # if device.type == "cuda": // weight loading fails if this check is active
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    # load weights
    weights = os.path.join(
        MODEL_WEIGHTS_DIR,
        mode,
        f"{mode}_{model_selection}.tar",
    )

    checkpoint = torch.load(weights, map_location="cpu")

    if not os.path.exists(weights):
        raise NotImplementedError(
            f"No weights found for model {mode} and selection {model_selection}")

    model.load_state_dict(checkpoint["model_state"])

    return model


def _apply_test_time_augmentations(data: Dict, inferer: SlidingWindowInferer, model: torch.nn.Module):
    n = 1.0
    for _ in range(4):
        # test time augmentations
        _img = RandGaussianNoised(keys="images", prob=1.0, std=0.001)(data)[
            "images"
        ]

        output = inferer(_img, model)
        outputs = outputs + output
        n = n + 1.0
        for dims in [[2], [3]]:
            flip_pred = inferer(torch.flip(_img, dims=dims), model)

            output = torch.flip(flip_pred, dims=dims)
            outputs = outputs + output
            n = n + 1.0
    outputs = outputs / n
    return outputs


# TODO refactor!
def _create_nifti_seg(
    threshold,
    reference_file,
    onehot_model_outputs_CHWD,
    output_file,
    whole_network_output_file,
    enhancing_network_output_file,
):
    # generate segmentation nifti
    activated_outputs = (
        (onehot_model_outputs_CHWD[0][:, :, :,
         :].sigmoid()).detach().cpu().numpy()
    )

    binarized_outputs = activated_outputs >= threshold

    binarized_outputs = binarized_outputs.astype(np.uint8)

    whole_metastasis = binarized_outputs[0]
    enhancing_metastasis = binarized_outputs[1]

    final_seg = whole_metastasis.copy()
    final_seg[whole_metastasis == 1] = 1  # edema
    final_seg[enhancing_metastasis == 1] = 2  # enhancing

    # get header and affine from T1
    REF = nib.load(reference_file)

    segmentation_image = nib.Nifti1Image(final_seg, REF.affine, REF.header)
    nib.save(segmentation_image, output_file)

    if whole_network_output_file:
        whole_network_output_file = Path(
            os.path.abspath(whole_network_output_file))

        whole_out = binarized_outputs[0]

        whole_out_image = nib.Nifti1Image(whole_out, REF.affine, REF.header)
        nib.save(whole_out_image, whole_network_output_file)

    if enhancing_network_output_file:
        enhancing_network_output_file = Path(
            os.path.abspath(enhancing_network_output_file)
        )

        enhancing_out = binarized_outputs[1]

        enhancing_out_image = nib.Nifti1Image(
            enhancing_out, REF.affine, REF.header)
        nib.save(enhancing_out_image, enhancing_network_output_file)
