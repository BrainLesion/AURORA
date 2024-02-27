from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List

import monai
import nibabel as nib
import numpy as np
from brainles_aurora.inferer.config import AuroraInfererConfig
from brainles_aurora.inferer.constants import IMGS_TO_MODE_DICT, DataMode, InferenceMode
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

logger = logging.getLogger(__name__)


class DataHandler:
    """Class to perform data related tasks such as validation, loading, transformation, saving."""

    def __init__(self, config: AuroraInfererConfig) -> "DataHandler":
        self.config = config
        # Following will be inferred from the input images during validation
        self.input_mode = None
        self.num_input_modalities = None
        self.reference_nifti_file = None

    def get_input_mode(self) -> DataMode:
        """Get the input mode.

        Returns:
            DataMode: Input mode.
        Raises:
            AssertionError: If the input mode is not set (i.e. input images were not validated)
        """
        assert (
            self.input_mode is not None
        ), "Input mode not set. Please validate the input images first by calling .validate_images(...)."
        return self.input_mode

    def get_num_input_modalities(self) -> int:
        """Get the number of input modalities.

        Returns:
            int: Number of input modalities.
        Raises:

            AssertionError: If the number of input modalities is not set (i.e. input images were not validated)
        """
        assert (
            self.num_input_modalities is not None
        ), "Number of input modalities not set. Please validate the input images first by calling .validate_images(...)."
        return self.num_input_modalities

    def get_reference_nifti_file(self) -> Path | str:
        """Get a reference NIfTI file from the input (first not None in order T1-T1C-T2-FLAIR) to match header and affine.

        Returns:
            Path: Path to reference NIfTI file.
        Raises:
            AssertionError: If the reference NIfTI file is not set (i.e. input images were not validated)
        """
        assert (
            self.reference_nifti_file is not None
        ), "Reference NIfTI file not set. Please ensure you provided paths to NIfTI images and validated the input images first by calling .validate_images(...)."
        return self.reference_nifti_file

    def validate_images(
        self,
        t1: str | Path | np.ndarray | None = None,
        t1c: str | Path | np.ndarray | None = None,
        t2: str | Path | np.ndarray | None = None,
        fla: str | Path | np.ndarray | None = None,
    ) -> List[np.ndarray | None] | List[Path | None]:
        """Validate the input images. \n
        Verify that the input images exist (for paths) and are all of the same type (NumPy or NIfTI).
        Sets internal variables input_mode, num_input_modalities and reference_nifti_file.

        Args:
            t1 (str | Path | np.ndarray | None, optional): T1 modality. Defaults to None.
            t1c (str | Path | np.ndarray | None, optional): T1C modality. Defaults to None.
            t2 (str | Path | np.ndarray | None, optional): T2 modality. Defaults to None.
            fla (str | Path | np.ndarray | None, optional): FLAIR modality. Defaults to None.

        Returns:
            List[np.ndarray | None] | List[Path | None]: List of validated images.
        Raises:
            FileNotFoundError: If a file is not found.
        Raises:
            ValueError: If a file path is not a NIfTI file (.nii or .nii.gz).
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
        self.num_input_modalities = len(not_none_images)
        if self.input_mode is DataMode.NIFTI_FILE:
            self.reference_nifti_file = not_none_images[0]
        logger.info(
            f"Successfully validated input images (received {self.num_input_modalities}). Input mode: {self.input_mode}"
        )
        return images

    def determine_inference_mode(
        self, images: List[np.ndarray | None] | List[Path | None]
    ) -> InferenceMode:
        """Determine the inference mode based on the provided images.
        Args:
            images (List[np.ndarray | None] | List[Path | None]): List of validated images.
        Returns:
            InferenceMode: Inference mode based on the combination of input images.
        Raises:
            NotImplementedError: If no model is implemented for the combination of input images.
        Raises:
            AssertionError: If the input mode is not set (i.e. input images were not validated)
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
                f"No model implemented for this combination of images: T1: {_t1}, T1C: {_t1c}, T2: {_t2}, FLAIR: {_flair}. {os.linesep}Available models: {[mode.value for mode in InferenceMode]}"
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
        Raises:
            AssertionError: If the input mode is not set (i.e. input images were not validated)
        """
        assert (
            self.input_mode is not None
        ), "Input mode not set. Please validate the input images first by calling .validate_images(...)."
        filtered_images = [img for img in images if img is not None]
        # init transforms
        transforms = [
            (
                LoadImageD(keys=["images"], image_only=True)
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

    def save_as_nifti(
        self, postproc_data: Dict[str, np.ndarray], output_file_mapping: Dict[str, str]
    ) -> None:
        """Save post-processed data as NIFTI files.

        Args:
            postproc_data (Dict[str, np.ndarray]): Post-processed data.
            output_file_mapping (Dict[str,str]): Mapping of output keys to output file paths.
        """
        # determine affine/ header
        if self.get_input_mode() == DataMode.NIFTI_FILE:
            reference_file = self.get_reference_nifti_file()
            ref = nib.load(reference_file)
            affine, header = ref.affine, ref.header
        else:
            logger.warning(
                f"Writing NIFTI output after NumPy input, using default affine=np.eye(4) and header=None"
            )
            affine, header = np.eye(4), None
        # save NIfTI files
        for key, data in postproc_data.items():
            output_file = output_file_mapping[key]
            if output_file:
                output_image = nib.Nifti1Image(data, affine, header)
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                nib.save(output_image, output_file)
                logger.info(f"Saved {key} to {output_file}")
