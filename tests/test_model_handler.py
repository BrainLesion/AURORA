import pytest
from brainles_aurora.inferer.model import ModelHandler


class TestModelHandler:
    def mock_model_handler(self, mock_config):
        return ModelHandler(config=mock_config)


# TODO: add tests
