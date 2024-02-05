import logging
from logging import Logger
import os
import json
from abc import ABC, abstractmethod
from pathlib import Path
import sys
from typing import Dict, List
import signal
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
import traceback

from brainles_aurora.inferer import (
    IMGS_TO_MODE_DICT,
    DataMode,
    InferenceMode,
    Output,
    AuroraInfererConfig,
    BaseConfig,
)
from brainles_aurora.utils import (
    download_model_weights,
    remove_path_suffixes,
)

from auxiliary.turbopath import turbopath

logger = logging.getLogger(__name__)


class AbstractInferer(ABC):
    """
    Abstract base class for inference.

    Attributes:
        config (BaseConfig): The configuration for the inferer.
        output_folder (Path): The output folder for the inferer. Follows the schema {config.output_folder}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
    """

    def __init__(self, config: BaseConfig) -> None:
        """Initialize the abstract inferer

        Args:
            config (BaseConfig): Configuration for the inferer.
        """
        self.config = config
        self._setup_logger()

        # download weights if not present
        self.lib_path: str = Path(os.path.dirname(os.path.abspath(__file__)))

        self.model_weights_folder = self.lib_path.parent / "model_weights"
        if not self.model_weights_folder.exists():
            download_model_weights(target_folder=str(self.lib_path.parent))

    def _set_log_file(self, log_file: str | Path) -> None:
        """Set the log file for the inference run and remove the file handler from a potential previous run.

        Args:
            log_file (str | Path): log file path
        """
        if self.log_file_handler:
            logging.getLogger().removeHandler(self.log_file_handler)

        parent_dir = os.path.dirname(log_file)
        # create parent dir if the path is more than just a file name
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        self.log_file_handler = logging.FileHandler(log_file)
        self.log_file_handler.setFormatter(
            logging.Formatter(
                "[%(levelname)s|%(module)s|L%(lineno)d] %(asctime)s: %(message)s",
                "%Y-%m-%dT%H:%M:%S%z",
            )
        )

        # Add the file handler to the !root! logger
        logging.getLogger().addHandler(self.log_file_handler)

    def _setup_logger(self) -> Logger:
        """Setup the logger for the inferer and overwrite system hooks to add logging for exceptions and signals."""
        config_file = Path(__file__).parent / "log_config.json"
        with open(config_file) as f_in:
            log_config = json.load(f_in)
        logging.config.dictConfig(log_config)
        logging.basicConfig(level=self.config.log_level)
        self.log_file_handler = None

        # overwrite system hooks to log exceptions and signals (SIGINT, SIGTERM)
        #! NOTE: This will note work in Jupyter Notebooks, (Without extra setup) see https://stackoverflow.com/a/70469055:
        def exception_handler(exception_type, value, tb):
            """Handle exceptions

            Args:
                exception_type (Exception): Exception type
                exception (Exception): Exception
                traceback (Traceback): Traceback
            """
            logger.error("".join(traceback.format_exception(exception_type, value, tb)))

            if issubclass(exception_type, SystemExit):
                # add specific code if exception was a system exit
                sys.exit(value.code)

        def signal_handler(sig, frame):
            signame = signal.Signals(sig).name
            logger.error(f"Received signal {sig} ({signame}), exiting...")
            sys.exit(0)

        sys.excepthook = exception_handler

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

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
        super().__init__(config=config)

        logger.info(f"Initialized {self.__class__.__name__} with config: {self.config}")
        self.device = self._configure_device()

        self.validated_images = None
        self.inference_mode = None

    def _validate_images(
        self,
        t1: str | Path | np.ndarray | None = None,
        t1c: str | Path | np.ndarray | None = None,
        t2: str | Path | np.ndarray | None = None,
        fla: str | Path | np.ndarray | None = None,
    ) -> List[np.ndarray | None] | List[Path | None]:
        """Validate the input images. \n
        Verify that the input images exist (for paths) and are all of the same type (NumPy or Nifti).
        If the input is a numpy array, the input mode is set to DataMode.NUMPY, otherwise to DataMode.NIFTI_FILE.

        Args:
            t1 (str | Path | np.ndarray | None, optional): T1 modality. Defaults to None.
            t1c (str | Path | np.ndarray | None, optional): T1C modality. Defaults to None.
            t2 (str | Path | np.ndarray | None, optional): T2 modality. Defaults to None.
            fla (str | Path | np.ndarray | None, optional): FLAIR modality. Defaults to None.

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
            return Path(turbopath(data))

        images = [
            _validate_image(img)
            for img in [
                t1,
                t1c,
                t2,
                fla,
            ]
        ]

        not_none_images = [img for img in images if img is not None]
        assert len(not_none_images) > 0, "No input images provided"
        # make sure all inputs have the same type
        unique_types = set(map(type, not_none_images))
        assert (
            len(unique_types) == 1
        ), f"All passed images must be of the same type! Received {unique_types}. Accepted Input types: {list(DataMode)}"

        logger.info(
            f"Successfully validated input images. Input mode: {self.input_mode}"
        )
        return images

    def _determine_inference_mode(
        self, images: List[np.ndarray | None] | List[Path | None]
    ) -> InferenceMode:
        """Determine the inference mode based on the provided images.

        Args:
            images (List[np.ndarray | None] | List[Path | None]): List of validated images.

        Raises:
            NotImplementedError: If no model is implemented for the combination of input images.
        Returns:
            InferenceMode: Inference mode based on the combination of input images.
        """
        _t1, _t1c, _t2, _fla = [img is not None for img in images]
        logger.info(f"Received files: T1: {_t1}, T1C: {_t1c}, T2: {_t2}, FLAIR: {_fla}")

        # check if files are given in a valid combination that has an existing model implementation
        mode = IMGS_TO_MODE_DICT.get((_t1, _t1c, _t2, _fla), None)

        if mode is None:
            raise NotImplementedError(
                "No model implemented for this combination of images"
            )

        logger.info(f"Inference mode: {mode}")
        return mode

    def _get_data_loader(self) -> torch.utils.data.DataLoader:
        """Get the data loader for inference.

        Returns:
            torch.utils.data.DataLoader: Data loader for inference.
        """
        # init transforms
        transforms = [
            (
                LoadImageD(keys=["images"])
                if self.input_mode == DataMode.NIFTI_FILE
                else None
            ),
            (
                EnsureChannelFirstd(keys="images")
                if (
                    len(self._get_not_none_files()) == 1
                    and self.input_mode == DataMode.NIFTI_FILE
                )
                else None
            ),
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
            "images": self._get_not_none_files(),
        }

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
            self.model_weights_folder,
            self.inference_mode,
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
        assert self.validated_images is not None, "Images not validated yet"

        return [img for img in self.validated_images if img is not None]

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
            logger.warning(
                f"Writing NIFTI output after NumPy input, using default affine=np.eye(4) and header=None"
            )
            affine, header = np.eye(4), None

        # save niftis
        for key, data in postproc_data.items():
            output_file = self.output_file_mapping[key]
            if output_file:
                output_image = nib.Nifti1Image(data, affine, header)
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                nib.save(output_image, output_file)
                logger.info(f"Saved {key} to {output_file}")

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
        return {
            Output.SEGMENTATION: final_seg,
            Output.WHOLE_NETWORK: whole_out,
            Output.METASTASIS_NETWORK: enhancing_out,
        }

    def _sliding_window_inference(self) -> Dict[str, np.ndarray]:
        """Perform sliding window inference using monai.inferers.SlidingWindowInferer.

        Returns:
            Dict[str, np.ndarray]: Post-processed data
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
            # currently always only 1 batch! TODO: potentialy add support to pass multiple image tuples at once?
            for data in self.data_loader:
                inputs = data["images"].to(self.device)

                outputs = inferer(inputs, self.model)
                if self.config.tta:
                    logger.info("Applying test time augmentations")
                    outputs = self._apply_test_time_augmentations(
                        outputs, data, inferer
                    )

                logger.info("Post-processing data")
                postprocessed_data = self._post_process(
                    onehot_model_outputs_CHWD=outputs,
                )

                # save data to fie if paths are provided
                if any(self.output_file_mapping.values()):
                    logger.info("Saving post-processed data as NIFTI files")
                    self._save_as_nifti(postproc_data=postprocessed_data)

                logger.info("Returning post-processed data as Dict of Numpy arrays")
                return postprocessed_data

    def _configure_device(self) -> torch.device:
        """Configure the device for inference.

        Returns:
            torch.device: Configured device.
        """
        device = torch.device("cpu")
        logger.info(f"Using device: {device}")
        return device

    def infer(
        self,
        t1: str | Path | np.ndarray | None = None,
        t1c: str | Path | np.ndarray | None = None,
        t2: str | Path | np.ndarray | None = None,
        fla: str | Path | np.ndarray | None = None,
        segmentation_file: str | Path | None = None,
        whole_tumor_unbinarized_floats_file: str | Path | None = None,
        metastasis_unbinarized_floats_file: str | Path | None = None,
        log_file: str | Path | None = None,
    ) -> Dict[str, np.ndarray]:
        """Perform inference on the provided images.

        Args:
            t1 (str | Path | np.ndarray | None, optional): T1 modality. Defaults to None.
            t1c (str | Path | np.ndarray | None, optional): T1C modality. Defaults to None.
            t2 (str | Path | np.ndarray | None, optional): T2 modality. Defaults to None.
            fla (str | Path | np.ndarray | None, optional): FLAIR modality. Defaults to None.
            segmentation_file (str | Path | None, optional): Path where the segementation file should be stored. Defaults to None. Should be a nifti file. Defaults internally to a './segmentation.nii.gz'.
            whole_tumor_unbinarized_floats_file (str | Path | None, optional): Path. Defaults to None.
            metastasis_unbinarized_floats_file (str | Path | None, optional): _description_. Defaults to None.
            log_file (str | Path | None, optional): _description_. Defaults to o the same path as segmentation_file with the extension .log or to ./{self.__class__.__name__}.log if no segmentation_file is provided.

        Returns:
            Dict[str, np.ndarray]: Post-processed data.
        """
        # setup log file for inference run
        if log_file:
            self._set_log_file(log_file=log_file)
        else:
            # if no log file is provided: set logfile to segmentation filename if provided, else inferer class name
            self._set_log_file(
                log_file=(
                    remove_path_suffixes(segmentation_file).with_suffix(".log")
                    if segmentation_file
                    else os.path.abspath(f"./{self.__class__.__name__}.log")
                ),
            )
        logger.info(f"Infer with config: {self.config} and device: {self.device}")

        # check inputs and get mode , if mode == prev mode => run inference, else load new model
        prev_mode = self.inference_mode
        self.validated_images = self._validate_images(t1=t1, t1c=t1c, t2=t2, fla=fla)
        self.inference_mode = self._determine_inference_mode(
            images=self.validated_images
        )
        if prev_mode != self.inference_mode:
            logger.info("No loaded compatible model found. Loading Model and weights")
            self.model = self._get_model()
        else:
            logger.info(
                f"Same inference mode {self.inference_mode} as previous infer call. Re-using loaded model"
            )
        # self.model.eval()
        logger.info("Setting up Dataloader")
        self.data_loader = self._get_data_loader()

        # setup output file paths
        self.output_file_mapping = {
            Output.SEGMENTATION: segmentation_file,
            Output.WHOLE_NETWORK: whole_tumor_unbinarized_floats_file,
            Output.METASTASIS_NETWORK: metastasis_unbinarized_floats_file,
        }

        ########
        logger.info(f"Running inference on device := {self.device}")
        out = self._sliding_window_inference()
        logger.info(f"Finished inference {os.linesep}")
        return out


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
        logger.info(f"Set torch device: {device}")

        # clean memory
        torch.cuda.empty_cache()
        return device
