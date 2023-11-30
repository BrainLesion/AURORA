# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import unittest

import nibabel as nib
import numpy as np
import logging
from brainles_aurora.inferer import AuroraInferer, AuroraGPUInferer


class TestInferer(unittest.TestCase):

    def setUp(self) -> None:
        self.t1 = "example_data/BraTS-MET-00110-000-t1n.nii.gz"
        self.t1c = "example_data/BraTS-MET-00110-000-t1c.nii.gz"
        self.t2 = "example_data/BraTS-MET-00110-000-t2w.nii.gz"
        self.fla = "example_data/BraTS-MET-00110-000-t2f.nii.gz"

    def load_np_from_nifti(self, path: str) -> np.ndarray:
        return nib.load(path).get_fdata()

    def test_invalid_inference_mode(self):
        """Might change with new models, rather a dummy test"""
        with self.assertRaises(NotImplementedError):
            inferer = AuroraInferer(
                #    t1=self.t1,
                t1c=self.t1c,
                t2=self.t2,
                fla=self.fla
            )
            inferer.infer(output_file="your_segmentation_file.nii.gz")

    def test_mixed_input_types(self):
        with self.assertRaises(AssertionError):
            inferer = AuroraInferer(
                t1=self.t1,
                t1c=self.load_np_from_nifti(self.t1c),
            )
            inferer.infer(output_file="your_segmentation_file.nii.gz")
   
    def test_gpu(self):
        # with self.assertRaises(AssertionError):
        inferer = AuroraGPUInferer(
            t1=self.t1,
            t1c=self.t1c,
        )
        inferer.infer(output_file="your_segmentation_file.nii.gz")
