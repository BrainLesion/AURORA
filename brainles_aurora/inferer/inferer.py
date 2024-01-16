import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List

import monai
import nibabel as nib
import numpy as np
import torch
from monai.data import list_data_collate
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import BasicUNet
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    Lambdad,
    LoadImageD,
    RandGaussianNoised,
    ScaleIntensityRangePercentilesd,
    ToTensord,
)
from torch.utils.data import DataLoader

from brainles_aurora.inferer.constants import IMGS_TO_MODE_DICT, DataMode, InferenceMode
from brainles_aurora.aux import turbo_path
from brainles_aurora.inferer.dataclasses import AuroraInfererConfig, BaseConfig
from brainles_aurora.inferer.download import download_model_weights

LIB_ABSPATH: str = os.path.dirname(os.path.abspath(__file__))

MODEL_WEIGHTS_DIR = Path(LIB_ABSPATH).parent / "model_weights"
if not MODEL_WEIGHTS_DIR.exists():
    download_model_weights(target_folder=LIB_ABSPATH)


class AbstractInferer(ABC):
    """
    Abstract base class for inference.

    Attributes:
        config (BaseConfig): The configuration for the inferer.
        output_folder (Path): The output folder for the inferer. Follows the schema {config.output_folder}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
    """

    def __init__(self, config: BaseConfig) -> None:
        """Initialize the abstract inferer. Sets up the logger and output folder.

        Args:
            config (BaseConfig): Configuration for the inferer.
        """
        self.config = config

        # setup output folder
        self.output_folder = (
            Path(os.path.abspath(self.config.output_folder))
            / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )
        self.output_folder.mkdir(exist_ok=True, parents=True)

        # setup logger
        self._setup_logger()

    def _setup_logger(self) -> None:
        """Set up the logger for the inferer."""

        self.log_path = self.output_folder / f"{self.config.segmentation_file_name}.log"

        logging.basicConfig(
            # stream=sys.stderr,
            format="%(asctime)s %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=self.config.log_level,
            encoding="utf-8",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.log_path),
            ],
        )

        class CustomStdErrStream:
            """Capture stderr and log it to the logger."""

            def write(self, msg: str):
                if msg := msg.rstrip():
                    logging.error(msg)

        # sys.stderr = CustomStdErrStream()

        logging.info(f"Logging to: {self.log_path}")

    @abstractmethod
    def infer(self):
        pass


class AuroraInferer(AbstractInferer):
    """Inferer for the CPU Aurora models."""

    def __init__(self, config: AuroraInfererConfig) -> None:
        """Initialize the AuroraInferer.

        Args:
            config (AuroraInfererConfig): Configuration for the Aurora inferer.
        """
        # TODO move weights path / download to config and setup
        super().__init__(config=config)

        logging.info(
            f"Initialized {self.__class__.__name__} with config: {self.config}"
        )

        self.images = self._validate_images()
        self.mode = self._determine_inference_mode()

        self.device = self._configure_device()
        logging.info("Setting up Dataloader")
        self.data_loader = self._get_data_loader()
        logging.info("Loading Model and weights")
        self.model = self._get_model()

    def _validate_images(self) -> List[np.ndarray | None] | List[Path | None]:
        """Validate input images, sets the input mode and returns the list of validated images.

        Returns:
            List[np.ndarray | None] | List[Path | None]: List of validated images.
        """

        def _validate_image(
            data: str | Path | np.ndarray | None,
        ) -> np.ndarray | Path | None:
            if data is None:
                return None
            if isinstance(data, np.ndarray):
                self.input_mode = DataMode.NUMPY
                return data.astype(np.float32)
            if not os.path.exists(data):
                raise FileNotFoundError(f"File {data} not found")
            if not (data.endswith(".nii.gz") or data.endswith(".nii")):
                raise ValueError(
                    f"File {data} must be a nifti file with extension .nii or .nii.gz"
                )
            self.input_mode = DataMode.NIFTI_FILE
            return turbo_path(data)

        images = [
            _validate_image(img)
            for img in [
                self.config.t1,
                self.config.t1c,
                self.config.t2,
                self.config.fla,
            ]
        ]

        not_none_images = [img for img in images if img is not None]
        assert len(not_none_images) > 0, "No input images provided"
        # make sure all inputs have the same type
        unique_types = set(map(type, not_none_images))
        assert (
            len(unique_types) == 1
        ), f"All passed images must be of the same type! Received {unique_types}. Accepted Input types: {list(DataMode)}"

        logging.info(
            f"Successfully validated input images. Input mode: {self.input_mode}"
        )
        return images

    def _determine_inference_mode(self) -> InferenceMode:
        """Determine the inference mode based on the provided images.

        Raises:
            NotImplementedError: If no model is implemented for the combination of input images.
        Returns:
            InferenceMode: Inference mode based on the combination of input images.
        """
        _t1, _t1c, _t2, _fla = [img is not None for img in self.images]
        logging.info(
            f"Received files: T1: {_t1}, T1C: {_t1c}, T2: {_t2}, FLAIR: {_fla}"
        )

        # check if files are given in a valid combination that has an existing model implementation
        mode = IMGS_TO_MODE_DICT.get((_t1, _t1c, _t2, _fla), None)

        if mode is None:
            raise NotImplementedError(
                "No model implemented for this combination of images"
            )

        logging.info(f"Inference mode: {mode}")
        return mode

    def _get_data_loader(self) -> torch.utils.data.DataLoader:
        """Get the data loader for inference.

        Returns:
            torch.utils.data.DataLoader: Data loader for inference.
        """
        # init transforms
        transforms = [
            LoadImageD(keys=["images"])
            if self.input_mode == DataMode.NIFTI_FILE
            else None,
            EnsureChannelFirstd(keys="images")
            if len(self._get_not_none_files()) == 1
            else None,
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
        data = {
            key: getattr(self.config, key)
            for key in ["t1", "t1c", "t2", "fla"]
            if getattr(self.config, key) is not None
        }
        # method returns files in standard order T1 T1C T2 FLAIR
        data["images"] = self._get_not_none_files()

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
        """Get the Aurora model based on the inference mode.

        Returns:
            torch.nn.Module: Aurora model.
        """

        # fuckery:
        x = 70 / 0
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
                f"No weights found for model {self.mode} and selection {self.config.model_selection}"
            )

        model = model.to(self.device)
        checkpoint = torch.load(weights_path, map_location=self.device)

        # The models were trained using DataParallel, hence we need to remove the 'module.' prefix
        # for cpu inference to enable checkpoint loading (since DataParallel is not usable for CPU)
        if self.device == torch.device("cpu"):
            if "module." in list(checkpoint["model_state"].keys())[0]:
                checkpoint["model_state"] = {
                    k.replace("module.", ""): v
                    for k, v in checkpoint["model_state"].items()
                }
        else:
            model = torch.nn.parallel.DataParallel(model)

        model.load_state_dict(checkpoint["model_state"])

        return model

    def _apply_test_time_augmentations(
        self, outputs: torch.Tensor, data: Dict, inferer: SlidingWindowInferer
    ) -> torch.Tensor:
        """Apply test time augmentations to the model outputs.

        Args:
            outputs (torch.Tensor): Model outputs.
            data (Dict): Input data.
            inferer (SlidingWindowInferer): Sliding window inferer.

        Returns:
            torch.Tensor: Augmented model outputs.
        """
        n = 1.0
        for _ in range(4):
            # test time augmentations
            _img = RandGaussianNoised(keys="images", prob=1.0, std=0.001)(data)[
                "images"
            ]

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
        """Get the list of non-None input images in  order T1-T1C-T2-FLA.

        Returns:
            List[np.ndarray] | List[Path]: List of non-None images.
        """
        return [img for img in self.images if img is not None]

    def _save_as_nifti(self, postproc_data: Dict[str, np.ndarray]) -> None:
        """Save post-processed data as NIFTI files.

        Args:
            postproc_data (Dict[str, np.ndarray]): Post-processed data.
        """
        # determine affine/ header
        if self.input_mode == DataMode.NIFTI_FILE:
            reference_file = self._get_not_none_files()[0]
            ref = nib.load(reference_file)
            affine, header = ref.affine, ref.header
        else:
            logging.warning(
                f"Writing NIFTI output after NumPy input, using default affine=np.eye(4) and header=None"
            )
            affine, header = np.eye(4), None

        logging.info(f"Output folder set to {self.output_folder}")

        # save niftis
        for key, data in postproc_data.items():
            # TODO: verify and make enum?
            if key == "segmentation":
                output_file = self.output_folder / self.config.segmentation_file_name
            else:
                output_file = self.output_folder / f"{key}.nii.gz"
            output_image = nib.Nifti1Image(data, affine, header)
            nib.save(output_image, output_file)
            logging.info(f"Saved {key} to {output_file}")

    def _post_process(
        self, onehot_model_outputs_CHWD: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """Post-process the model outputs.

        Args:
            onehot_model_outputs_CHWD (torch.Tensor): One-hot encoded model outputs.

        Returns:
            Dict[str, np.ndarray]: Post-processed data.
        """

        # create segmentations
        activated_outputs = (
            (onehot_model_outputs_CHWD[0][:, :, :, :].sigmoid()).detach().cpu().numpy()
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

        # create output dict based on config
        data = {"segmentation": final_seg}
        if self.config.output_whole_network:
            data["output_whole_network"] = whole_out
        if self.config.output_metastasis_network:
            data["output_metastasis_network"] = enhancing_out
        return data

    def _sliding_window_inference(self) -> None | Dict[str, np.ndarray]:
        """Perform sliding window inference using monai.inferers.SlidingWindowInferer.

        Returns:
            None | Dict[str, np.ndarray]: Post-processed data if output_mode is NUMPY, otherwise the data is saved as a niftis and None is returned.
        """
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
            # loop through batches, only 1 batch!
            for data in self.data_loader:
                inputs = data["images"].to(self.device)

                outputs = inferer(inputs, self.model)
                if self.config.tta:
                    logging.info("Applying test time augmentations")
                    outputs = self._apply_test_time_augmentations(
                        outputs, data, inferer
                    )

                postprocessed_data = self._post_process(
                    onehot_model_outputs_CHWD=outputs,
                )
                if self.config.output_mode == DataMode.NUMPY:
                    return postprocessed_data
                else:
                    self._save_as_nifti(postproc_data=postprocessed_data)
                    return

    def _configure_device(self) -> torch.device:
        """Configure the device for inference.

        Returns:
            torch.device: Configured device.
        """
        device = torch.device("cpu")
        logging.info(f"Using device: {device}")
        return device

    def infer(self) -> None:
        """Run the inference process."""
        logging.info(f"Running inference on {self.device}")
        return self._sliding_window_inference()


####################
# GPU Inferer
####################
class AuroraGPUInferer(AuroraInferer):
    """Inferer for the Aurora models on GPU."""

    def __init__(
        self,
        config: AuroraInfererConfig,
        cuda_devices: str = "0",
    ) -> None:
        """Initialize the AuroraGPUInferer.

        Args:
            config (AuroraInfererConfig): Configuration for the Aurora GPU inferer.
            cuda_devices (str, optional): CUDA devices to use. Defaults to "0".
        """
        self.cuda_devices = cuda_devices

        super().__init__(config=config)

    def _configure_device(self) -> torch.device:
        """Configure the GPU device for inference.

        Returns:
            torch.device: Configured GPU device.
        """
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_devices

        assert (
            torch.cuda.is_available()
        ), "No cuda device available while using GPUInferer"

        device = torch.device("cuda")
        logging.info(f"Using device: {device}")

        # clean memory
        torch.cuda.empty_cache()
        return device
