from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from brainles_aurora.inferer.config import AuroraInfererConfig
from brainles_aurora.inferer.constants import InferenceMode
from brainles_aurora.inferer.data import DataHandler


class TestDataHandler:
    @pytest.fixture
    def t1_path(self):
        return "example/data/BraTS-MET-00110-000-t1n.nii.gz"

    @pytest.fixture
    def t1c_path(self):
        return "example/data/BraTS-MET-00110-000-t1c.nii.gz"

    @pytest.fixture
    def t2_path(self):
        return "example/data/BraTS-MET-00110-000-t2w.nii.gz"

    @pytest.fixture
    def fla_path(self):
        return "example/data/BraTS-MET-00110-000-t2f.nii.gz"

    @pytest.fixture
    def mock_config(self):
        return AuroraInfererConfig()

    @pytest.fixture
    def mock_data_handler(self, mock_config):
        return DataHandler(config=mock_config)

    @pytest.fixture
    def load_np_from_nifti(self):
        def _load_np_from_nifti(path):
            return nib.load(path).get_fdata()

        return _load_np_from_nifti

    def test_validate_images_numpy(
        self,
        t1_path,
        t1c_path,
        t2_path,
        fla_path,
        mock_data_handler,
        load_np_from_nifti,
    ):
        images = mock_data_handler.validate_images(
            t1=load_np_from_nifti(t1_path),
            t1c=load_np_from_nifti(t1c_path),
            t2=load_np_from_nifti(t2_path),
            fla=load_np_from_nifti(fla_path),
        )
        assert len(images) == 4
        assert all(isinstance(img, np.ndarray) for img in images)

    def test_validate_images_nifti(
        self,
        t1_path,
        t1c_path,
        t2_path,
        fla_path,
        mock_data_handler,
    ):
        images = mock_data_handler.validate_images(
            t1=t1_path,
            t1c=t1c_path,
            t2=t2_path,
            fla=fla_path,
        )
        print(images)
        assert len(images) == 4
        assert all(isinstance(img, Path) for img in images)

    def test_validate_images_file_not_found(
        self,
        mock_data_handler,
    ):
        with pytest.raises(FileNotFoundError):
            _ = mock_data_handler.validate_images(t1="invalid_path.nii.gz")

    def test_validate_images_different_types(
        self,
        mock_data_handler,
        t1_path,
        t1c_path,
        load_np_from_nifti,
    ):
        with pytest.raises(AssertionError):
            _ = mock_data_handler.validate_images(
                t1=t1_path, t1c=load_np_from_nifti(t1c_path)
            )

    def test_validate_images_no_inputs(
        self,
        mock_data_handler,
    ):
        with pytest.raises(AssertionError):
            _ = mock_data_handler.validate_images()

    def test_determine_inference_mode(
        self,
        mock_data_handler,
        t1_path,
    ):
        validated_images = mock_data_handler.validate_images(t1=t1_path)
        mode = mock_data_handler.determine_inference_mode(images=validated_images)
        assert isinstance(mode, InferenceMode)

    def test_determine_inference_mode_not_implemented(
        self,
        mock_data_handler,
        t2_path,
    ):
        images = mock_data_handler.validate_images(t2=t2_path)
        with pytest.raises(NotImplementedError):
            _ = mock_data_handler.determine_inference_mode(images=images)
