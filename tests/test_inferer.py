from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest
import torch

from brainles_aurora.inferer.constants import InferenceMode
from brainles_aurora.inferer.dataclasses import AuroraInfererConfig
from brainles_aurora.inferer.inferer import AuroraInferer, AuroraGPUInferer


class TestAuroraInferer:
    @pytest.fixture
    def t1_path(self):
        return "example_data/BraTS-MET-00110-000-t1n.nii.gz"

    @pytest.fixture
    def t1c_path(self):
        return "example_data/BraTS-MET-00110-000-t1c.nii.gz"

    @pytest.fixture
    def t2_path(self):
        return "example_data/BraTS-MET-00110-000-t2w.nii.gz"

    @pytest.fixture
    def fla_path(self):
        return "example_data/BraTS-MET-00110-000-t2f.nii.gz"

    @pytest.fixture
    def mock_config(self, t1_path, t1c_path, t2_path, fla_path):
        return AuroraInfererConfig(t1=t1_path, t1c=t1c_path, t2=t2_path, fla=fla_path)

    @pytest.fixture
    def mock_inferer(self, mock_config):
        return AuroraInferer(config=mock_config)

    @pytest.fixture
    def load_np_from_nifti(self):
        def _load_np_from_nifti(path):
            return nib.load(path).get_fdata()

        return _load_np_from_nifti

    def test_validate_images(self, mock_config):
        inferer = AuroraInferer(config=mock_config)
        images = inferer._validate_images()
        assert len(images) == 4
        assert all(isinstance(img, Path) for img in images)

    def test_validate_images_file_not_found(self, mock_config):
        mock_config.t1 = "invalid_path.nii.gz"
        with pytest.raises(FileNotFoundError):
            _ = AuroraInferer(config=mock_config)
            # called internally in __init__
            # inferer._validate_images()

    def test_validate_images_different_types(self, mock_config, load_np_from_nifti):
        mock_config.t1 = load_np_from_nifti(mock_config.t1)
        with pytest.raises(AssertionError):
            _ = AuroraInferer(config=mock_config)
            # called internally in __init__
            # inferer._validate_images()

    def test_validate_images_no_inputs(self, mock_config, load_np_from_nifti):
        mock_config.t1 = None
        mock_config.t1c = None
        mock_config.t2 = None
        mock_config.fla = None
        with pytest.raises(AssertionError):
            _ = AuroraInferer(config=mock_config)
            # called internally in __init__
            # inferer._validate_images()

    def test_determine_inference_mode(self, mock_config):
        inferer = AuroraInferer(config=mock_config)
        mode = inferer._determine_inference_mode()
        assert isinstance(mode, InferenceMode)

    def test_determine_inference_mode_not_implemented(self, mock_config):
        mock_validated_images = [
            None,
            None,
            None,
            None,
        ]  # set all to None to raise NotImplementedError
        with pytest.raises(NotImplementedError), patch(
            "brainles_aurora.inferer.inferer.AuroraInferer._validate_images",
            return_value=mock_validated_images,
        ):
            inferer = AuroraInferer(config=mock_config)
            # called internally in __init__
            # inferer._determine_inference_mode()

    def test_get_data_loader(self, mock_config):
        inferer = AuroraInferer(config=mock_config)
        data_loader = inferer._get_data_loader()
        assert isinstance(data_loader, torch.utils.data.DataLoader)

    def test_get_model(self, mock_config):
        inferer = AuroraInferer(config=mock_config)
        model = inferer._get_model()
        assert isinstance(model, torch.nn.Module)

    def test_setup_logger(self, mock_config):
        inferer = AuroraInferer(config=mock_config)
        assert inferer.log_path is not None
        assert inferer.output_folder.exists()
        assert inferer.output_folder.is_dir()

    def test_infer(self, mock_config):
        inferer = AuroraInferer(config=mock_config)
        with patch.object(inferer, "_sliding_window_inference", return_value=None):
            inferer.infer()

    def test_configure_device(self, mock_config):
        inferer = AuroraInferer(config=mock_config)
        device = inferer._configure_device()
        assert device == torch.device("cpu")

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Skipping GPU device test since cuda is not available",
    )
    def test_configure_device_gpu(self, mock_config):
        inferer = AuroraGPUInferer(config=mock_config)
        device = inferer._configure_device()
        assert device == torch.device("cuda")

    def test_get_model(self, mock_config):
        inferer = AuroraInferer(config=mock_config)
        model = inferer._get_model()
        assert isinstance(model, torch.nn.Module)

    def test_get_data_loader(self, mock_config):
        inferer = AuroraInferer(config=mock_config)
        data_loader = inferer._get_data_loader()
        assert isinstance(data_loader, torch.utils.data.DataLoader)
