from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import pytest
import torch

from brainles_aurora.inferer import (
    InferenceMode,
    AuroraInfererConfig,
    AuroraInferer,
    AuroraGPUInferer,
)


class TestAuroraInferer:
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
    def mock_inferer(self, mock_config):
        return AuroraInferer(config=mock_config)

    @pytest.fixture
    def load_np_from_nifti(self):
        def _load_np_from_nifti(path):
            return nib.load(path).get_fdata()

        return _load_np_from_nifti

    def test_validate_images(
        self,
        t1_path,
        t1c_path,
        t2_path,
        fla_path,
        mock_inferer,
    ):
        images = mock_inferer._validate_images(
            t1=t1_path,
            t1c=t1c_path,
            t2=t2_path,
            fla=fla_path,
        )
        assert len(images) == 4
        assert all(isinstance(img, Path) for img in images)

    def test_validate_images_file_not_found(
        self,
        mock_inferer,
    ):
        with pytest.raises(FileNotFoundError):
            _ = mock_inferer._validate_images(t1="invalid_path.nii.gz")

    def test_validate_images_different_types(
        self,
        mock_inferer,
        t1_path,
        t1c_path,
        load_np_from_nifti,
    ):
        with pytest.raises(AssertionError):
            _ = mock_inferer._validate_images(
                t1=t1_path, t1c=load_np_from_nifti(t1c_path)
            )

    def test_validate_images_no_inputs(
        self,
        mock_inferer,
    ):
        with pytest.raises(AssertionError):
            _ = mock_inferer._validate_images()

    def test_determine_inference_mode(
        self,
        mock_inferer,
        t1_path,
    ):
        validated_images = mock_inferer._validate_images(t1=t1_path)
        mode = mock_inferer._determine_inference_mode(images=validated_images)
        assert isinstance(mode, InferenceMode)

    def test_determine_inference_mode_not_implemented(
        self,
        mock_inferer,
        t2_path,
    ):
        images = mock_inferer._validate_images(t2=t2_path)
        with pytest.raises(NotImplementedError):
            mode = mock_inferer._determine_inference_mode(images=images)

    def test_infer(
        self,
        mock_inferer,
        t1_path,
    ):
        with patch.object(mock_inferer, "_sliding_window_inference", return_value=None):
            mock_inferer.infer(t1=t1_path)

    def test_configure_device(
        self,
        mock_config,
    ):
        inferer = AuroraInferer(config=mock_config)
        device = inferer._configure_device()
        assert device == torch.device("cpu")

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Skipping GPU device test since cuda is not available",
    )
    def test_configure_device_gpu(
        self,
        mock_config,
    ):
        inferer = AuroraGPUInferer(config=mock_config)
        device = inferer._configure_device()
        assert device == torch.device("cuda")
