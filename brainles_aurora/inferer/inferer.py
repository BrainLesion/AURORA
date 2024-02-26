import json
import logging
import os
import signal
import sys
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np
import torch
from brainles_aurora.inferer import (
    IMGS_TO_MODE_DICT,
    AuroraInfererConfig,
    BaseConfig,
    DataMode,
    Device,
    Output,
)
from brainles_aurora.inferer.data import DataHandler
from brainles_aurora.utils import download_model_weights, remove_path_suffixes
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import BasicUNet
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class AbstractInferer(ABC):
    """
    Abstract base class for inference.
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

    def _setup_logger(self):
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
    """Inferer for the Aurora models."""

    def __init__(self, config: AuroraInfererConfig) -> None:
        """Initialize the AuroraInferer.

        Args:
            config (AuroraInfererConfig): Configuration for the Aurora inferer.
        """
        super().__init__(config=config)
        logger.info(f"Initialized {self.__class__.__name__} with config: {self.config}")
        self.device = self._configure_device()

        self.inference_mode = None
        self.data_handler = DataHandler(config=self.config)
        # self.model_handler = ModelHandler(config=self.config)

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
            outputs += output
            n += 1.0
            for dims in [[2], [3]]:
                flip_pred = inferer(torch.flip(_img, dims=dims), self.model)

                output = torch.flip(flip_pred, dims=dims)
                outputs += output
                n += 1.0
        outputs /= n
        return outputs

    def _save_as_nifti(self, postproc_data: Dict[str, np.ndarray]) -> None:
        """Save post-processed data as NIFTI files.

        Args:
            postproc_data (Dict[str, np.ndarray]): Post-processed data.
        """
        # determine affine/ header
        if self.data_handler.get_input_mode() == DataMode.NIFTI_FILE:
            reference_file = self.data_handler.get_reference_nifti_file()
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

    def _sliding_window_inference(
        self, data_loader: DataLoader
    ) -> Dict[str, np.ndarray]:
        """Perform sliding window inference using monai.inferers.SlidingWindowInferer.

        Args:
            data_loader (DataLoader): Data loader.

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
            for data in data_loader:
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
        """Configure the device for inference based on the specified config.device.

        Returns:
            torch.device: Configured device.
        """
        if self.config.device == Device.CPU:
            device = torch.device("cpu")
        if self.config.device == Device.AUTO or self.config.device == Device.GPU:
            if torch.cuda.is_available():
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                os.environ["CUDA_VISIBLE_DEVICES"] = self.config.cuda_devices
                # clean memory
                torch.cuda.empty_cache()
                device = torch.device("cuda")
            else:
                if self.config.device == Device.GPU:
                    logger.warning(
                        "Requested GPU device, but no CUDA devices available. Falling back to CPU."
                    )
                device = torch.device("cpu")

        logger.info(f"Set torch device: {device}")

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
        validated_images = self.data_handler.validate_images(
            t1=t1, t1c=t1c, t2=t2, fla=fla
        )
        determined_inference_mode = self.data_handler.determine_inference_mode(
            images=validated_images
        )

        logger.info("Setting up Dataloader")
        data_loader = self.data_handler.get_data_loader(images=validated_images)

        if self.inference_mode != determined_inference_mode:
            logger.info(
                f"No loaded compatible model found (Switching from {self.inference_mode} to {determined_inference_mode}). Loading Model and weights"
            )
            self.inference_mode = determined_inference_mode
            self.model = self._get_model()
        else:
            logger.info(
                f"Same inference mode ({self.inference_mode}) as previous infer call. Re-using loaded model"
            )

        # setup output file paths
        self.output_file_mapping = {
            Output.SEGMENTATION: segmentation_file,
            Output.WHOLE_NETWORK: whole_tumor_unbinarized_floats_file,
            Output.METASTASIS_NETWORK: metastasis_unbinarized_floats_file,
        }

        logger.info(f"Running inference on device := {self.device}")
        out = self._sliding_window_inference(data_loader=data_loader)
        logger.info(f"Finished inference {os.linesep}")
        return out
