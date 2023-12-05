from unittest.mock import Mock, patch

import nibabel as nib
import pytest
import torch

from brainles_aurora.inferer import (AbstractInferer, AuroraInferer,
                                     AuroraInfererConfig, BaseConfig)


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
        return AuroraInfererConfig(
            t1=t1_path, t1c=t1c_path, t2=t2_path, fla=fla_path
        )

    @pytest.fixture
    def mock_inferer(self, mock_config):
        return AuroraInferer(config=mock_config)

    @pytest.fixture
    def load_np_from_nifti(self, request):
        def _load_np_from_nifti(path):
            return nib.load(path).get_fdata()

        return _load_np_from_nifti

    def test_no_inputs(self):
        """Might change with new models, rather a dummy test"""
        with pytest.raises(AssertionError):
            config = AuroraInfererConfig()
            _ = AuroraInferer(config=config)

    # def test_invalid_inference_mode(t1c_path, t2_path, fla_path):
    #     """Might change with new models, rather a dummy test"""
    #     with pytest.raises(NotImplementedError):
    #         config = AuroraInfererConfig(t1c=t1c_path, t2=t2_path, fla=fla_path)
    #         _ = AuroraInferer(config=config)

    def test_mixed_input_types(self, t1_path, t1c_path, load_np_from_nifti):
        with pytest.raises(AssertionError):
            config = AuroraInfererConfig(
                t1=t1_path, t1c=load_np_from_nifti(t1c_path))
            _ = AuroraInferer(config=config)

    def test_setup_logger(self, mock_config):
        inferer = AuroraInferer(config=mock_config)
        assert inferer.log_path is not None
        assert inferer.output_folder.exists()
        assert inferer.output_folder.is_dir()

    def test_infer(self, mock_config):
        inferer = AuroraInferer(config=mock_config)
        with patch.object(inferer, '_sliding_window_inference', return_value=None):
            inferer.infer()

    def test_configure_device(self, mock_config):
        inferer = AuroraInferer(config=mock_config)
        device = inferer._configure_device()
        assert device == torch.device("cpu")

    # def test_post_process(self, mock_config):
    #     inferer = AuroraInferer(config=mock_config)
    #     mock_outputs = Mock()
    #     result = inferer._post_process(onehot_model_outputs_CHWD=mock_outputs)
    #     assert isinstance(result, dict)

    # def test_sliding_window_inference(self, mock_config):
    #     inferer = AuroraInferer(config=mock_config)
    #     with patch.object(inferer, '_post_process', return_value=None):
    #         inferer._sliding_window_inference()

    def test_get_model(self, mock_config):
        inferer = AuroraInferer(config=mock_config)
        model = inferer._get_model()
        assert isinstance(model, torch.nn.Module)

    def test_get_data_loader(self, mock_config):
        inferer = AuroraInferer(config=mock_config)
        data_loader = inferer._get_data_loader()
        assert isinstance(data_loader, torch.utils.data.DataLoader)

    def test_determine_inference_mode(self, mock_config):
        inferer = AuroraInferer(config=mock_config)
        mode = inferer._determine_inference_mode()
        assert mode is not None

    def test_validate_images_valid(self, mock_config):
        inferer = AuroraInferer(config=mock_config)
        images = inferer._validate_images()
        assert len(images) == 4
