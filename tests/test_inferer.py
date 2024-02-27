from unittest.mock import patch

import pytest
import torch
from brainles_aurora.inferer.config import AuroraInfererConfig
from brainles_aurora.inferer.constants import Device
from brainles_aurora.inferer.inferer import AuroraInferer


class TestAuroraInferer:
    @pytest.fixture
    def t1_path(self):
        return "example/data/BraTS-MET-00110-000-t1n.nii.gz"

    @pytest.fixture
    def mock_inferer(self):
        return AuroraInferer()

    def test_infer(
        self,
        mock_inferer,
        t1_path,
    ):
        with patch.object(
            mock_inferer.model_handler, "_sliding_window_inference", return_value=None
        ):
            mock_inferer.infer(t1=t1_path)

    def test_configure_device_cpu(self):
        inferer = AuroraInferer(config=AuroraInfererConfig(device=Device.CPU))
        device = inferer._configure_device()
        assert device == torch.device("cpu")

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Skipping GPU device test since cuda is not available",
    )
    def test_configure_device_gpu(
        self,
    ):
        inferer = AuroraInferer()
        device = inferer._configure_device()
        assert device == torch.device("cuda")
