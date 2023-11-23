
import logging
import os
from abc import ABC, abstractmethod
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
from brainles_aurora.constants import (IMGS_TO_MODE_DICT, InputMode, InferenceMode,
                                       ModelSelection)
from brainles_aurora.download import download_model_weights

LIB_ABSPATH: str = os.path.dirname(os.path.abspath(__file__))

MODEL_WEIGHTS_DIR = os.path.join(LIB_ABSPATH, "model_weights")
if not os.path.exists(MODEL_WEIGHTS_DIR):
    download_model_weights(target_folder=LIB_ABSPATH)


class AuroraInferer(ABC):

    def __init__(self,
                 segmentation_file: str | np.ndarray,
                 t1: str | Path | np.ndarray | None = None,
                 t1c: str | Path | np.ndarray | None = None,
                 t2: str | Path | np.ndarray | None = None,
                 fla: str | Path | np.ndarray | None = None,
                 tta: bool = True,
                 sliding_window_batch_size: int = 1,
                 workers: int = 0,
                 threshold: float = 0.5,
                 sliding_window_overlap: float = 0.5,
                 crop_size: Tuple[int, int, int] = (192, 192, 32),
                 model_selection: ModelSelection = ModelSelection.BEST,
                 whole_network_outputs_file: str | None = None,
                 metastasis_network_outputs_file: str | None = None,
                 log_level: int | str = logging.INFO,
                 ) -> None:
        self.segmentation_file = segmentation_file
        self.t1 = t1
        self.t1c = t1c
        self.t2 = t2
        self.fla = fla
        self.tta = tta
        self.sliding_window_batch_size = sliding_window_batch_size
        self.workers = workers
        self.threshold = threshold
        self.sliding_window_overlap = sliding_window_overlap
        self.crop_size = crop_size
        self.model_selection = model_selection
        self.whole_network_outputs_file = whole_network_outputs_file
        self.metastasis_network_outputs_file = metastasis_network_outputs_file
        self.log_level = log_level

        # setup
        self._setup_logger()
        self.images = self._validate_images()
        self.mode = self._determine_inferece_mode()

    def _setup_logger(self):
        logging.basicConfig(
            format='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            level=self.log_level,
            encoding='utf-8',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('aurora_inferer.log')
            ]
        )

    def _validate_images(self):
        def _validate_img(data: str | Path | np.ndarray | None) -> np.ndarray | Path | None:
            if data is None:
                return None
            if isinstance(data, np.ndarray):
                self.input_mode = InputMode.NUMPY
                return data.astype(np.float32)
            if not os.path.exists(data):
                raise FileNotFoundError(f"File {data} not found")
            if not (data.endswith(".nii.gz") or data.endswith(".nii")):
                raise ValueError(
                    f"File {data} must be a nifti file with extension .nii or .nii.gz")
            self.input_mode = InputMode.FILE
            return turbo_path(data)

        images = [_validate_img(img)
                  for img in [self.t1, self.t1c, self.t2, self.fla]]

        assert len(set(map(type, [img for img in images if img is not None]))
                   ) == 1, f"All passed images must be of the same type! Accepted Input types: {list(InputMode)}"
        logging.info(
            f"Successfully validated input images. Input mode: {self.input_mode}")
        return images

    def _determine_inferece_mode(self) -> InferenceMode:

        _t1, _t1c, _t2, _fla = [img is not None for img in self.images]
        logging.info(
            f"Received files: T1: {_t1}, T1C: {_t1c}, T2: {_t2}, FLAIR: {_fla}"
        )

        # check if files are given in a valid combination that has an existing model implementation
        mode = IMGS_TO_MODE_DICT.get((_t1, _t1c, _t2, _fla), None)

        if mode is None:
            raise NotImplementedError(
                "No model implemented for this combination of images")

        logging.info(f"Inference mode based on passed images: {mode}")
        return mode

    def _get_data_loader(self) -> torch.utils.data.DataLoader:
        # init transforms
        transforms = [
            LoadImageD(keys=["images"]
                       ) if self.input_mode == InputMode.FILE else None,
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
        # filter None transforms taht may be included due to conditions
        transforms = list(filter(None, transforms))
        inference_transforms = Compose(transforms)

        # init data dictionary
        data = {}
        if self.t1 is not None:
            data['t1'] = self.t1
        if self.t1c is not None:
            data['t1c'] = self.t1c
        if self.t2 is not None:
            data['t2'] = self.t2
        if self.fla is not None:
            data['fla'] = self.fla
        # method returns files in standard order T1 T1C T2 FLAIR
        data['images'] = self._get_not_none_files()

        #  instantiate dataset and dataloader
        infererence_ds = monai.data.Dataset(
            data=[data],
            transform=inference_transforms,
        )

        data_loader = DataLoader(
            infererence_ds,
            batch_size=1,
            num_workers=self.workers,
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

        model = torch.nn.DataParallel(model)
        model = model.to(self.device)

        # load weights
        weights = os.path.join(
            MODEL_WEIGHTS_DIR,
            self.mode,
            f"{self.model_selection}.tar",
        )

        if not os.path.exists(weights):
            raise NotImplementedError(
                f"No weights found for model {self.mode} and selection {self.model_selection}")

        checkpoint = torch.load(weights, map_location=self.device)
        model.load_state_dict(checkpoint["model_state"])

        return model

    def _apply_test_time_augmentations(self, data: Dict, inferer: SlidingWindowInferer):
        n = 1.0
        for _ in range(4):
            # test time augmentations
            _img = RandGaussianNoised(keys="images", prob=1.0, std=0.001)(data)[
                "images"
            ]

            output = inferer(_img, self.model)
            outputs = outputs + output
            n = n + 1.0
            for dims in [[2], [3]]:
                flip_pred = inferer(torch.flip(_img, dims=dims), self.model)

                output = torch.flip(flip_pred, dims=dims)
                outputs = outputs + output
                n = n + 1.0
        outputs = outputs / n
        return outputs

    def _get_not_none_files(self) -> List[np.ndarray]:
        return [img for img in self.images if img is not None]

    def _create_nifti_seg(self,
                          reference_file: str | Path,
                          onehot_model_outputs_CHWD):
        # generate segmentation nifti
        activated_outputs = (
            (onehot_model_outputs_CHWD[0][:, :, :,
                                          :].sigmoid()).detach().cpu().numpy()
        )

        binarized_outputs = activated_outputs >= self.threshold

        binarized_outputs = binarized_outputs.astype(np.uint8)

        whole_metastasis = binarized_outputs[0]
        enhancing_metastasis = binarized_outputs[1]

        final_seg = whole_metastasis.copy()
        final_seg[whole_metastasis == 1] = 1  # edema
        final_seg[enhancing_metastasis == 1] = 2  # enhancing

        # get header and affine from reference
        ref = nib.load(reference_file)

        segmentation_image = nib.Nifti1Image(final_seg, ref.affine, ref.header)
        nib.save(segmentation_image, self.output_file)

        if whole_network_output_file:
            whole_network_output_file = Path(
                os.path.abspath(whole_network_output_file))

            whole_out = binarized_outputs[0]

            whole_out_image = nib.Nifti1Image(
                whole_out, ref.affine, ref.header)
            nib.save(whole_out_image, whole_network_output_file)

        if enhancing_network_output_file:
            enhancing_network_output_file = Path(
                os.path.abspath(enhancing_network_output_file)
            )

            enhancing_out = binarized_outputs[1]

            enhancing_out_image = nib.Nifti1Image(
                enhancing_out, ref.affine, ref.header)
            nib.save(enhancing_out_image, enhancing_network_output_file)

    def _sliding_window_inference(self):
        inferer = SlidingWindowInferer(
            roi_size=self.crop_size,  # = patch_size
            sw_batch_size=self.sliding_window_batch_size,
            sw_device=self.device,
            device=self.device,
            overlap=self.sliding_window_overlap,
            mode="gaussian",
            padding_mode="replicate",
        )

        with torch.no_grad():
            self.model.eval()
            # loop through batches
            for data in tqdm(self.data_loader, 0):
                inputs = data["images"]

                outputs = inferer(inputs, self.model)
                if self.tta:
                    outputs = self._apply_test_time_augmentations(
                        data, inferer
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

                self._create_nifti_seg(
                    reference_file=reference_file,
                    onehot_model_outputs_CHWD=outputs,
                )

    def infer(self):
        logging.info("Loading data and model")
        self.data_loader = self._get_data_loader()
        self.model = self._get_model()

        logging.info(f"Running inference on {self.device}")
        return self._infer()

    @abstractmethod
    def _configure_device(self) -> torch.device:
        pass

    @abstractmethod
    def _infer(self):
        pass


####################
# GPU Inferer
####################
class GPUInferer(AuroraInferer):

    def __init__(self,
                 segmentation_file: str | np.ndarray,
                 t1: str | Path | np.ndarray | None = None,
                 t1c: str | Path | np.ndarray | None = None,
                 t2: str | Path | np.ndarray | None = None,
                 fla: str | Path | np.ndarray | None = None,
                 cuda_devices: str = "0",
                 tta: bool = True,
                 sliding_window_batch_size: int = 1,
                 workers: int = 0,
                 threshold: float = 0.5,
                 sliding_window_overlap: float = 0.5,
                 crop_size: Tuple[int, int, int] = (192, 192, 32),
                 model_selection: ModelSelection = ModelSelection.BEST,
                 whole_network_outputs_file: str | None = None,
                 metastasis_network_outputs_file: str | None = None,
                 log_level: int | str = logging.INFO,
                 ) -> None:
        super().__init__(
            segmentation_file=segmentation_file,
            t1=t1,
            t1c=t1c,
            t2=t2,
            fla=fla,
            tta=tta,
            sliding_window_batch_size=sliding_window_batch_size,
            workers=workers,
            threshold=threshold,
            sliding_window_overlap=sliding_window_overlap,
            crop_size=crop_size,
            model_selection=model_selection,
            whole_network_outputs_file=whole_network_outputs_file,
            metastasis_network_outputs_file=metastasis_network_outputs_file,
            log_level=log_level,
        )
        # GPUInferer specific variables
        self.cuda_devices = cuda_devices

        self.device = self._configure_device()

    def _infer(self):
        self._sliding_window_inference()

    def _configure_device(self) -> torch.device:

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_devices

        assert torch.cuda.is_available(), "No cuda device available while using GPUInferer"

        device = torch.device("cuda")
        logging.info(f"Using device: {device}")

        # clean memory
        torch.cuda.empty_cache()
        return device

####################
# CPU Inferer
####################


class CPUInferer(AuroraInferer):

    def __init__(self,
                 segmentation_file: str | np.ndarray,
                 t1: str | Path | np.ndarray | None = None,
                 t1c: str | Path | np.ndarray | None = None,
                 t2: str | Path | np.ndarray | None = None,
                 fla: str | Path | np.ndarray | None = None,
                 tta: bool = True,
                 sliding_window_batch_size: int = 1,
                 workers: int = 0,
                 threshold: float = 0.5,
                 sliding_window_overlap: float = 0.5,
                 crop_size: Tuple[int, int, int] = (192, 192, 32),
                 model_selection: ModelSelection = ModelSelection.BEST,
                 whole_network_outputs_file: str | None = None,
                 metastasis_network_outputs_file: str | None = None,
                 log_level: int | str = logging.INFO,
                 ) -> None:
        super().__init__(
            segmentation_file=segmentation_file,
            t1=t1,
            t1c=t1c,
            t2=t2,
            fla=fla,
            tta=tta,
            sliding_window_batch_size=sliding_window_batch_size,
            workers=workers,
            threshold=threshold,
            sliding_window_overlap=sliding_window_overlap,
            crop_size=crop_size,
            model_selection=model_selection,
            whole_network_outputs_file=whole_network_outputs_file,
            metastasis_network_outputs_file=metastasis_network_outputs_file,
            log_level=log_level,
        )
        self.device = self._configure_device()

    def _infer(self):
        self._sliding_window_inference()

    def _configure_device(self) -> torch.device:
        device = torch.device("cpu")
        logging.info(f"Using device: {device}")

        return device
