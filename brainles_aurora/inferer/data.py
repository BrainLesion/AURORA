import logging
import os
from pathlib import Path
from typing import List

import monai
import numpy as np
from monai.data import list_data_collate
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    Lambdad,
    LoadImageD,
    ScaleIntensityRangePercentilesd,
    ToTensord,
)
from torch.utils.data import DataLoader

from brainles_aurora.inferer.config import AuroraInfererConfig
from brainles_aurora.inferer.constants import IMGS_TO_MODE_DICT, DataMode, InferenceMode

logger = logging.getLogger(__name__)


class DataHandler:
    """DataHandler class for handling input images and creating the data loader for inference."""

    def __init__(self, config: AuroraInfererConfig) -> "DataHandler":
        self.config = config
        self.input_mode = (
            None  # Will be inferred from the input images during validation
        )

    def validate_images(
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
                    f"File {data} must be a NIfTI file with extension .nii or .nii.gz"
                )
            self.input_mode = DataMode.NIFTI_FILE
            return Path(data).absolute()

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

    def determine_inference_mode(
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

        assert (
            self.input_mode is not None
        ), "Please validate the input images first by calling validate_images(...)."

        _t1, _t1c, _t2, _flair = [img is not None for img in images]
        logger.info(
            f"Received files: T1: {_t1}, T1C: {_t1c}, T2: {_t2}, FLAIR: {_flair}"
        )

        # check if files are given in a valid combination that has an existing model implementation
        mode = IMGS_TO_MODE_DICT.get((_t1, _t1c, _t2, _flair), None)

        if mode is None:
            raise NotImplementedError(
                f"No model implemented for this combination of images: T1: {_t1}, T1C: {_t1c}, T2: {_t2}, FLAIR: {_flair}"
            )

        logger.info(f"Inference mode: {mode}")
        return mode

    def get_data_loader(
        self, images: List[np.ndarray | None] | List[Path | None]
    ) -> DataLoader:
        """Get the data loader for inference.

        Args:
            images (List[np.ndarray | None] | List[Path | None]): List of validated images.

        Returns:
            torch.utils.data.DataLoader: Data loader for inference.
        """

        assert (
            self.input_mode is not None
        ), "Input mode not set. Please validate the input images first by calling .validate_images(...)."

        filtered_images = [img for img in images if img is not None]
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
                    len(filtered_images) == 1 and self.input_mode == DataMode.NIFTI_FILE
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
            "images": filtered_images,
        }

        # init dataset and dataloader
        inference_ds = monai.data.Dataset(
            data=[data],
            transform=inference_transforms,
        )

        data_loader = DataLoader(
            inference_ds,
            batch_size=1,
            num_workers=self.config.workers,
            collate_fn=list_data_collate,
            shuffle=False,
        )
        return data_loader
