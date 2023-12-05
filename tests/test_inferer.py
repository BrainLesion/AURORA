import nibabel as nib
import pytest

from brainles_aurora.inferer import AuroraInferer, AuroraInfererConfig


@pytest.fixture
def t1_path():
    return "example_data/BraTS-MET-00110-000-t1n.nii.gz"


@pytest.fixture
def t1c_path():
    return "example_data/BraTS-MET-00110-000-t1c.nii.gz"


@pytest.fixture
def t2_path():
    return "example_data/BraTS-MET-00110-000-t2w.nii.gz"


@pytest.fixture
def fla_path():
    return "example_data/BraTS-MET-00110-000-t2f.nii.gz"


@pytest.fixture
def load_np_from_nifti(request):
    def _load_np_from_nifti(path):
        return nib.load(path).get_fdata()

    return _load_np_from_nifti


def test_no_inputs():
    """Might change with new models, rather a dummy test"""
    with pytest.raises(AssertionError):
        config = AuroraInfererConfig()
        _ = AuroraInferer(config=config)


def test_invalid_inference_mode(t1c_path, t2_path, fla_path):
    """Might change with new models, rather a dummy test"""
    with pytest.raises(NotImplementedError):
        config = AuroraInfererConfig(t1c=t1c_path, t2=t2_path, fla=fla_path)
        _ = AuroraInferer(config=config)


def test_mixed_input_types(t1_path, t1c_path, load_np_from_nifti):
    with pytest.raises(AssertionError):
        config = AuroraInfererConfig(
            t1=t1_path, t1c=load_np_from_nifti(t1c_path))
        _ = AuroraInferer(config=config)
