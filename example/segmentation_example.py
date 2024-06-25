from brainles_aurora.inferer import (
    AuroraInferer,
    AuroraInfererConfig,
)
import os
import logging
from path import Path
import nibabel as nib

BASE_PATH = Path(os.path.abspath(__file__)).parent.parent

t1 = BASE_PATH / "example/data/BraTS-MET-00110-000-t1n.nii.gz"
t1c = BASE_PATH / "example/data/BraTS-MET-00110-000-t1c.nii.gz"
t2 = BASE_PATH / "example/data/BraTS-MET-00110-000-t2w.nii.gz"
fla = BASE_PATH / "example/data/BraTS-MET-00110-000-t2f.nii.gz"


def load_np_from_nifti(path):
    return nib.load(path).get_fdata()


def gpu_nifti():
    config = AuroraInfererConfig(
        tta=False, cuda_devices="4"
    )  # disable tta for faster inference in this showcase

    # If you don-t have a GPU that supports CUDA use the CPU version: AuroraInferer(config=config)
    inferer = AuroraInferer(config=config)

    inferer.infer(
        t1=t1,
        t1c=t1c,
        t2=t2,
        fla=fla,
        segmentation_file="test_output/segmentation_tta.nii",
        whole_tumor_unbinarized_floats_file="test_output/whole_network_tta.nii.gz",
        metastasis_unbinarized_floats_file="test_output/metastasis_network_tta.nii.gz",
        log_file="test_output/custom_log.log",
    )


def gpu_nifti_2():
    config = AuroraInfererConfig(
        tta=False, cuda_devices="4"
    )  # disable tta for faster inference in this showcase

    # If you don-t have a GPU that supports CUDA use the CPU version: AuroraInferer(config=config)
    inferer = AuroraInferer(config=config)

    inferer.infer(
        t1=t1,
        segmentation_file="test_output/nevergonna_seg.nii.gz",
        whole_tumor_unbinarized_floats_file="test_output/whole_network.nii.gz",
        metastasis_unbinarized_floats_file="test_output/metastasis_network.nii.gz",
    )
    inferer.infer(
        t1=t1,
        segmentation_file="test_output2/randomseg.nii.gz",
    )


def gpu_np():
    config = AuroraInfererConfig(
        tta=False,
    )  # disable tta for faster inference in this showcase

    # If you don-t have a GPU that supports CUDA use the CPU version: AuroraInferer(config=config)
    inferer = AuroraInferer(config=config)

    t1_np = load_np_from_nifti(t1)
    inferer.infer(
        t1=t1_np,
    )


if __name__ == "__main__":
    # logging.basicConfig(level=)
    gpu_nifti()
    # gpu_nifti_2()
