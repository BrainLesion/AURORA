from __future__ import annotations

import json
import logging
import os
import signal
import sys
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from brainles_aurora.inferer.config import AuroraInfererConfig, BaseConfig
from brainles_aurora.inferer.constants import Device, Output
from brainles_aurora.inferer.data import DataHandler
from brainles_aurora.inferer.model import ModelHandler
from brainles_aurora.utils import remove_path_suffixes
from brainles_aurora.utils.console_decorators import citation_reminder

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
                "[%(levelname)-8s | %(module)-15s | L%(lineno)-5d] | %(asctime)s: %(message)s",
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

    def __init__(self, config: Optional[AuroraInfererConfig] = None) -> None:
        """Initialize the AuroraInferer.

        Args:
            config (Optional[AuroraInfererConfig]): Configuration for the inferer. If none provided, default config is used.
        """
        if not config:
            config = AuroraInfererConfig()
        super().__init__(config=config)
        logger.info(f"Initialized {self.__class__.__name__} with config: {self.config}")
        self.device = self._configure_device()
        self.data_handler = DataHandler(config=self.config)
        self.model_handler = ModelHandler(config=self.config, device=self.device)

    def _configure_device(self) -> torch.device:
        """Configure the device for inference based on the specified config.device.

        Returns:
            torch.device: Configured device.
        """
        if self.config.device == Device.CPU:
            device = torch.device("cpu")
        if self.config.device == Device.AUTO or self.config.device == Device.GPU:
            # The env vars have to be set ebfore the first call to torch.cuda, else torch will always attempt to use the first device
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = self.config.cuda_devices
            if torch.cuda.is_available():
                # clean memory
                torch.cuda.empty_cache()
                device = torch.device("cuda")
                logger.info(
                    f"Set CUDA_VISIBLE_DEVICES to {os.environ['CUDA_VISIBLE_DEVICES']}"
                )
                logger.debug(f"Available CUDA devices: {torch.cuda.device_count()}")
                logger.debug(f"Current CUDA devices: {torch.cuda.current_device()}")
            else:
                if self.config.device == Device.GPU:
                    logger.warning(
                        "Requested GPU device, but no CUDA devices available. Falling back to CPU."
                    )
                device = torch.device("cpu")

        logger.info(f"Set torch device: {device}")

        return device

    @citation_reminder
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
            whole_tumor_unbinarized_floats_file (str | Path | None, optional): Output path. Defaults to None.
            metastasis_unbinarized_floats_file (str | Path | None, optional): Output path. Defaults to None.
            log_file (str | Path | None, optional): Logfile path. Defaults to o the same path as segmentation_file with the extension .log or to ./{self.__class__.__name__}.log if no segmentation_file is provided.

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

        self.model_handler.load_model(
            inference_mode=determined_inference_mode,
            num_input_modalities=self.data_handler.get_num_input_modalities(),
        )

        logger.info("Setting up Dataloader")
        data_loader = self.data_handler.get_data_loader(images=validated_images)

        # setup output file paths
        output_file_mapping = {
            Output.SEGMENTATION: segmentation_file,
            Output.WHOLE_NETWORK: whole_tumor_unbinarized_floats_file,
            Output.METASTASIS_NETWORK: metastasis_unbinarized_floats_file,
        }

        logger.info(f"Running inference on device := {self.device}")
        out = self.model_handler.infer(data_loader=data_loader)
        logger.info(f"Finished inference")

        # save data to fie if paths are provided
        if any(output_file_mapping.values()):
            logger.info("Saving post-processed data as NIfTI files")
            self.data_handler.save_as_nifti(
                postproc_data=out, output_file_mapping=output_file_mapping
            )
        logger.info(f"{' Finished inference run ':=^80}")
        return out
