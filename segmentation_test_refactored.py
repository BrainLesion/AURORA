from brainles_aurora.core import infer

# infer(
#     t1c_file="example_data/BraTS-MET-00110-000-t1c.nii.gz",
#     segmentation_file="your_segmentation_file.nii.gz",
#     tta=False,  # optional: whether to use test time augmentations
#     verbosity=True,  # optional: verbosity of the output
# )


from brainles_aurora.inferer import CPUInferer

inf = CPUInferer(segmentation_file="your_segmentation_file.nii.gz",
                 t1_file="example_data/BraTS-MET-00110-000-t1n.nii.gz",
                 t1c_file="example_data/BraTS-MET-00110-000-t1c.nii.gz",)
inf.infer()
