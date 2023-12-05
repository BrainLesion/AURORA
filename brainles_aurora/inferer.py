
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import monai
import nibabel as nib
import numpy as np
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
from brainles_aurora.constants import (IMGS_TO_MODE_DICT, DataMode,
                                       InferenceMode, ModelSelection)
from brainles_aurora.download import download_model_weights

LIB_ABSPATH: str = os.path.dirname(os.path.abspath(__file__))

MODEL_WEIGHTS_DIR = Path(LIB_ABSPATH) / "model_weights"
if not os.path.exists(MODEL_WEIGHTS_DIR):
    download_model_weights(target_folder=LIB_ABSPATH)


@dataclass
class AuroraInfererConfig:
    t1: str | Path | np.ndarray | None = None
    t1c: str | Path | np.ndarray | None = None
    t2: str | Path | np.ndarray | None = None
    fla: str | Path | np.ndarray | None = None
    tta: bool = True
    sliding_window_batch_size: int = 1
    workers: int = 0
    threshold: float = 0.5
    sliding_window_overlap: float = 0.5
    crop_size: Tuple[int, int, int] = (192, 192, 32)
    model_selection: ModelSelection = ModelSelection.BEST
    whole_network_outputs_file: str | None = None
    metastasis_network_outputs_file: str | None = None
    log_level: int | str = logging.INFO


class AuroraInferer():

    def __init__(self, config: AuroraInfererConfig) -> None:
        self.config = config

        # setup
        self._setup_logger()

        logging.info(f"Initialized {self.__class__.__name__}")

        self.images = self._validate_images()
        self.mode = self._determine_inference_mode()

    def _setup_logger(self) -> None:
        logging.basicConfig(
            format='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            level=self.config.log_level,
            encoding='utf-8',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('aurora_inferer.log')
            ]
        )

    def _validate_images(self) -> List[np.ndarray | None] | List[Path | None]:
        def _validate_img(data: str | Path | np.ndarray | None) -> np.ndarray | Path | None:
            if data is None:
                return None
            if isinstance(data, np.ndarray):
                self.input_mode = DataMode.NUMPY
                return data.astype(np.float32)
            if not os.path.exists(data):
                raise FileNotFoundError(f"File {data} not found")
            if not (data.endswith(".nii.gz") or data.endswith(".nii")):
                raise ValueError(
                    f"File {data} must be a nifti file with extension .nii or .nii.gz")
            self.input_mode = DataMode.NIFTI_FILE
            return turbo_path(data)

        images = [_validate_img(img)
                  for img in [self.config.t1, self.config.t1c, self.config.t2, self.config.fla]]

        # make sure all inputs have the same type
        unique_types = set(filter(None, map(type, images)))
        assert len(
            unique_types) == 1, f"All passed images must be of the same type! Received {unique_types}. Accepted Input types: {list(DataMode)}"

        logging.info(
            f"Successfully validated input images. Input mode: {self.input_mode}")
        return images

    def _determine_inference_mode(self) -> InferenceMode:

        _t1, _t1c, _t2, _fla = [img is not None for img in self.images]
        logging.info(
            f"Received files: T1: {_t1}, T1C: {_t1c}, T2: {_t2}, FLAIR: {_fla}"
        )

        # check if files are given in a valid combination that has an existing model implementation
        mode = IMGS_TO_MODE_DICT.get((_t1, _t1c, _t2, _fla), None)

        if mode is None:
            raise NotImplementedError(
                "No model implemented for this combination of images")

        logging.info(f"Inference mode: {mode}")
        return mode

    def _get_data_loader(self) -> torch.utils.data.DataLoader:
        # init transforms
        transforms = [
            LoadImageD(keys=["images"]
                       ) if self.input_mode == DataMode.NIFTI_FILE else None,
            EnsureChannelFirstd(keys="images") if len(
                self._get_not_none_files()) == 1 else None,
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
        # Filter None transforms
        transforms = list(filter(None, transforms))
        inference_transforms = Compose(transforms)

        # Initialize data dictionary
        data = {key: getattr(self.config, key) for key in [
            "t1", "t1c", "t2", "fla"] if getattr(self.config, key) is not None}
        # method returns files in standard order T1 T1C T2 FLAIR
        data['images'] = self._get_not_none_files()

        # init dataset and dataloader
        infererence_ds = monai.data.Dataset(
            data=[data],
            transform=inference_transforms,
        )

        data_loader = DataLoader(
            infererence_ds,
            batch_size=1,
            num_workers=self.config.workers,
            collate_fn=list_data_collate,
            shuffle=False,
        )
        return data_loader

    def _get_model(self) -> torch.nn.Module:
        # init model
        model = BasicUNet(
            spatial_dims=3,
            in_channels=len(self._get_not_none_files()),
            out_channels=2,
            features=(32, 32, 64, 128, 256, 32),
            dropout=0.1,
            act="mish",
        )

        # load weights
        weights_path = os.path.join(
            MODEL_WEIGHTS_DIR,
            self.mode,
            f"{self.config.model_selection}.tar",
        )

        if not os.path.exists(weights_path):
            raise NotImplementedError(
                f"No weights found for model {self.mode} and selection {self.config.model_selection}")

        model = model.to(self.device)
        checkpoint = torch.load(weights_path, map_location=self.device)
        
        # The models were trained using dataparallel, so we need to remove the 'module.' prefix 
        # for cpu inference since DataParallel only works on GPU
        if self.device == torch.device("cpu"):
            if 'module.' in list(checkpoint["model_state"].keys())[0]:
                checkpoint["model_state"] = {k.replace('module.', ''): v for k, v in checkpoint["model_state"].items()}
        else:
            model = torch.nn.parallel.DataParallel(model)
        
        model.load_state_dict(checkpoint["model_state"])

        return model

    def _apply_test_time_augmentations(self, outputs: torch.Tensor, data: Dict, inferer: SlidingWindowInferer) -> torch.Tensor:
        n = 1.0
        for _ in range(4):
            # test time augmentations
            _img = RandGaussianNoised(
                keys="images", prob=1.0, std=0.001)(data)["images"]

            output = inferer(_img, self.model)
            outputs = outputs + output
            n += 1.0
            for dims in [[2], [3]]:
                flip_pred = inferer(torch.flip(_img, dims=dims), self.model)

                output = torch.flip(flip_pred, dims=dims)
                outputs = outputs + output
                n += 1.0
        outputs = outputs / n
        return outputs

    def _get_not_none_files(self) -> List[np.ndarray] | List[Path]:
        return [img for img in self.images if img is not None]

    def _save_output_nifti(self,
                           output: np.ndarray,
                           output_file: str | Path,
                           affine: np.ndarray,
                           header: nib.Nifti1Header | None) -> None:
        logging.info(
            f"Saving outputs to Nifti file {output_file}")
        output_file = Path(os.path.abspath(output_file))
        output_image = nib.Nifti1Image(output, affine, header)
        nib.save(output_image, output_file)

    def _save_output(self,
                     output_file: str | Path,
                     output_mode: DataMode,
                     reference_file: str | Path,
                     onehot_model_outputs_CHWD
                     ) -> None | Dict[str, np.ndarray]:
        logging.info(f"Saving output in mode: {output_mode}")

        # create segmentations
        activated_outputs = (
            (onehot_model_outputs_CHWD[0][:, :, :,
                                          :].sigmoid()).detach().cpu().numpy()
        )
        binarized_outputs = activated_outputs >= self.config.threshold
        binarized_outputs = binarized_outputs.astype(np.uint8)

        whole_metastasis = binarized_outputs[0]
        enhancing_metastasis = binarized_outputs[1]

        final_seg = whole_metastasis.copy()
        final_seg[whole_metastasis == 1] = 1  # edema
        final_seg[enhancing_metastasis == 1] = 2  # enhancing

        whole_out = binarized_outputs[0]
        enhancing_out = binarized_outputs[1]

        # save segmentations
        if output_mode == DataMode.NIFTI_FILE:
            if self.input_mode == DataMode.NIFTI_FILE:
                logging.info(
                    f"Saving segmentation to Nifti file {output_file} with affine/ header from reference file {reference_file}")
                ref = nib.load(reference_file)
                affine, header = ref.affine, ref.header
            else:
                logging.info(
                    f"Saving segmentation to Nifti file {output_file} with default affine np.exe(4) and None header")
                affine, header = np.eye(4), None

            segmentation_image = nib.Nifti1Image(
                final_seg, affine, header)
            nib.save(segmentation_image, output_file)
            self._save_output_nifti(
                final_seg, output_file, affine, header
            )

            if self.config.whole_network_outputs_file:
                self._save_output_nifti(
                    whole_out, self.config.whole_network_outputs_file, affine, header
                )

            if self.config.metastasis_network_outputs_file:
                self._save_output_nifti(
                    enhancing_out, self.config.metastasis_network_outputs_file, affine, header
                )
        else:
            # TODO: not sure if this is really desired
            raise NotImplementedError(
                "NumPy output mode not implemented yet!"
            )

    def _sliding_window_inference(self, output_file: str | Path, output_mode: DataMode) -> None:
        inferer = SlidingWindowInferer(
            roi_size=self.config.crop_size,  # = patch_size
            sw_batch_size=self.config.sliding_window_batch_size,
            sw_device=self.device,
            device=self.device,
            overlap=self.config.sliding_window_overlap,
            mode="gaussian",
            padding_mode="replicate",
        )

        with torch.no_grad():
            self.model.eval()
            self.model = self.model.to(self.device)
            # loop through batches
            for data in tqdm(self.data_loader, 0):
                inputs = data["images"].to(self.device)

                outputs = inferer(inputs, self.model)
                if self.config.tta:
                    logging.info("Applying test time augmentations")
                    outputs = self._apply_test_time_augmentations(
                        outputs, data, inferer
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
                        FileNotFoundError("No reference file found!")

                self._save_output(
                    output_file=output_file,
                    output_mode=output_mode,
                    reference_file=reference_file,
                    onehot_model_outputs_CHWD=outputs,
                )

    def infer(self, output_file: str | Path = "seg.nii.gz", output_mode: DataMode = DataMode.NIFTI_FILE) -> None:
        self.device = self._configure_device()
        logging.info("Setting up Dataloader")
        self.data_loader = self._get_data_loader()
        logging.info("Loading Model and weights")
        self.model = self._get_model()

        logging.info(f"Running inference on {self.device}")
        return self._infer(output_file, output_mode)

    def _configure_device(self) -> torch.device:
        device = torch.device("cpu")
        logging.info(f"Using device: {device}")
        return device

    def _infer(self, output_file: str | Path, output_mode: DataMode) -> None:
        return self._sliding_window_inference(output_file=output_file, output_mode=output_mode)


####################
# GPU Inferer
####################
class AuroraGPUInferer(AuroraInferer):

    def __init__(self,
                 config: AuroraInfererConfig,
                 cuda_devices: str = "0",
                 ) -> None:
        super().__init__(config=config)

        # GPUInferer specific variables
        self.cuda_devices = cuda_devices

    def _configure_device(self) -> torch.device:

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_devices

        assert torch.cuda.is_available(), "No cuda device available while using GPUInferer"

        device = torch.device("cuda")
        logging.info(f"Using device: {device}")

        # clean memory
        torch.cuda.empty_cache()
        return device
