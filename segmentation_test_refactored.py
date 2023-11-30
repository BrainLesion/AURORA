

from brainles_aurora.inferer import AuroraGPUInferer

t1 = "example_data/BraTS-MET-00110-000-t1n.nii.gz"
t1c = "example_data/BraTS-MET-00110-000-t1c.nii.gz"
t2 = "example_data/BraTS-MET-00110-000-t2w.nii.gz"
fla = "example_data/BraTS-MET-00110-000-t2f.nii.gz"

inferer = AuroraGPUInferer(
            t1=t1,
            t1c=t1c,
            t2=t2,
            fla=fla,
            metastasis_network_outputs_file="metastasis_network_outputs.nii.gz",
            whole_network_outputs_file="whole_network_outputs_file.nii.gz",
        )
inferer.infer(output_file="your_segmentation_file.nii.gz")
