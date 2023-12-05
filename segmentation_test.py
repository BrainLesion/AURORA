

from brainles_aurora.inferer import AuroraInferer, AuroraGPUInferer, AuroraInfererConfig
import os
from path import Path
import nibabel as nib

BASE_PATH = Path(os.path.abspath(__file__)).parent

t1 = BASE_PATH / "example_data/BraTS-MET-00110-000-t1n.nii.gz"
t1c = BASE_PATH / "example_data/BraTS-MET-00110-000-t1c.nii.gz"
t2 = BASE_PATH / "example_data/BraTS-MET-00110-000-t2w.nii.gz"
fla = BASE_PATH / "example_data/BraTS-MET-00110-000-t2f.nii.gz"


def load_np_from_nifti(path):
    return nib.load(path).get_fdata()


def gpu_nifti():
    config = AuroraInfererConfig(
        t1=t1,
        t1c=t1c,
        t2=t2,
        fla=fla,
    )
    inferer = AuroraGPUInferer(
        config=config,
    )
    inferer.infer()


def cpu_nifti():
    config = AuroraInfererConfig(
        t1=t1,
        t1c=t1c,
        t2=t2,
        fla=fla,
    )
    inferer = AuroraInferer(
        config=config,
    )
    inferer.infer()


def gpu_np():
    config = AuroraInfererConfig(
        t1=load_np_from_nifti(t1),
        t1c=load_np_from_nifti(t1c),
        t2=load_np_from_nifti(t2),
        fla=load_np_from_nifti(fla)
    )
    inferer = AuroraGPUInferer(
        config=config,
    )
    inferer.infer()


if __name__ == "__main__":
    gpu_nifti()
