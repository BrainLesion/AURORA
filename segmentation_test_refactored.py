

from brainles_aurora.inferer import AuroraInferer,AuroraGPUInferer, AuroraInfererConfig
import os
from path import Path
BASE_PATH = Path(os.path.abspath(__file__)).parent

t1 = BASE_PATH / "example_data/BraTS-MET-00110-000-t1n.nii.gz"
t1c = BASE_PATH / "example_data/BraTS-MET-00110-000-t1c.nii.gz"
t2 = BASE_PATH / "example_data/BraTS-MET-00110-000-t2w.nii.gz"
fla = BASE_PATH / "example_data/BraTS-MET-00110-000-t2f.nii.gz"

config = AuroraInfererConfig(
        t1=t1,
        t1c=t1c,
        t2=t2,
        fla=fla,
        metastasis_network_outputs_file="metastasis_network_outputs.nii.gz",
        whole_network_outputs_file="whole_network_outputs_file.nii.gz",
)
inferer = AuroraGPUInferer(
            config=config,
        )
inferer.infer(output_file="your_segmentation_file.nii.gz")
