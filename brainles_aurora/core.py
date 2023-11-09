from __future__ import annotations

import os

import torch
from aux import turbo_path
from enums import ModalityMode, ModelSelection
from path import Path

LIB_ABSPATH: str = os.path.dirname(os.path.abspath(__file__))


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
    model_selection="best",
        verbosity=True):
    # configure logger
    # transform inputs to paths
    t1_file, t1c_file, t2_file, fla_file = [turbo_path(
        f) for f in [t1_file, t1c_file, t2_file, fla_file] if f]
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
    # model
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


def _configure_device(cuda_devices: str) -> torch.device:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # clean memory
    torch.cuda.empty_cache()
    return device


def _determine_mode(t1_file,
                    t1c_file,
                    t2_file,
                    fla_file
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
    mode,
    t1_file,
    t1c_file,
    t2_file,
    fla_file,
    workers,
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
            "t1c": t1_file,
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

    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    # load weights
    weights = os.path.join(
        LIB_ABSPATH,
        "model_weights",
        mode.value,
        f"{mode.value}_{model_selection.value}.tar",
    )

    if not os.path.exists(weights):
        raise NotImplementedError(
            f"No weights found for model {mode.value} and selection {model_selection.value}")

    checkpoint = torch.load(model_weights, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])

    return model
