from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from brainles_aurora.inferer.config import AuroraInfererConfig
from brainles_aurora.inferer.constants import InferenceMode, Output, WEIGHTS_DIR_PATTERN
from brainles_aurora.utils import check_model_weights
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import BasicUNet
from monai.transforms import RandGaussianNoised
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class ModelHandler:
    """Class for model loading, inference and post processing"""

    def __init__(
        self, config: AuroraInfererConfig, device: torch.device
    ) -> "ModelHandler":
        """Initialize the ModelHandler and download model weights if necessary.

        Args:
            config (AuroraInfererConfig): config
            device (torch.device): torch device

        Returns:
            ModelHandler: ModelHandler instance
        """
        self.config = config
        self.device = device
        # Will be set during infer() call
        self.model = None
        self.inference_mode = None

        # get location of model weights
        self.model_weights_folder = check_model_weights()

    def load_model(
        self, inference_mode: InferenceMode, num_input_modalities: int
    ) -> None:
        """Load the model based on the inference mode. Will reuse previously loaded model if inference mode is the same.

        Args:
            inference_mode (InferenceMode): Inference mode
            num_input_modalities (int): Number of input modalities (range 1-4)
        """
        if not self.model or self.inference_mode != inference_mode:
            logger.info(
                f"No loaded compatible model found (Switching from {self.inference_mode} to {inference_mode}). Loading Model and weights..."
            )
            self.inference_mode = inference_mode
            self.model = self._load_model(num_input_modalities=num_input_modalities)
            logger.info(f"Successfully loaded model.")
        else:
            logger.info(
                f"Same inference mode ({self.inference_mode}) as previous infer call. Re-using loaded model"
            )

    def _load_model(self, num_input_modalities: int) -> torch.nn.Module:
        """Internal method to load the Aurora model based on the inference mode.
        Args:
            num_input_modalities (int): Number of input modalities (range 1-4)
        Returns:
            torch.nn.Module: Aurora model.
        """
        # init model
        model = BasicUNet(
            spatial_dims=3,
            in_channels=num_input_modalities,
            out_channels=2,
            features=(32, 32, 64, 128, 256, 32),
            dropout=0.1,
            act="mish",
        )
        # load weights
        weights_path = os.path.join(
            self.model_weights_folder,
            f"{self.inference_mode.value}_{self.config.model_selection.value}.tar",
        )
        if not os.path.exists(weights_path):
            raise NotImplementedError(
                f"No weights found for model {self.inference_mode} and selection {self.config.model_selection}. {os.linesep}Available models: {[mode.value for mode in InferenceMode]}"
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

    def _post_process(
        self, onehot_model_outputs_CHWD: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """Post-process the model outputs.

        Args:
            onehot_model_outputs_CHWD (torch.Tensor): One-hot encoded model outputs (Channel Height Width Depth).

        Returns:
            Dict[str, np.ndarray]: Post-processed data.
        """
        # create segmentations
        activated_outputs = (
            (onehot_model_outputs_CHWD[0][:, :, :, :].sigmoid()).detach().cpu().numpy()
        )
        binarized_outputs = activated_outputs >= self.config.threshold
        binarized_outputs = binarized_outputs.astype(np.uint8)
        # output channles
        whole_metastasis = binarized_outputs[0]
        enhancing_metastasis = binarized_outputs[1]
        # final seg
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
                logger.info("Returning post-processed data as Dict of Numpy arrays")
                return postprocessed_data

    def infer(self, data_loader: DataLoader) -> Dict[str, np.ndarray]:
        """Perform aurora inference on the given data_loader.

        Args:
            data_loader (DataLoader): data loader

        Returns:
            Dict[str, np.ndarray]: Post-processed data
        """
        return self._sliding_window_inference(data_loader=data_loader)
