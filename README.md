# AURORA

[![Python Versions](https://img.shields.io/pypi/pyversions/brainles_aurora)](https://pypi.org/project/brainles_aurora/)
[![Stable Version](https://img.shields.io/pypi/v/brainles_aurora?label=stable)](https://pypi.python.org/pypi/brainles_aurora/)
[![Documentation Status](https://readthedocs.org/projects/brainles-aurora/badge/?version=latest)](http://brainles-aurora.readthedocs.io/?badge=latest)
[![tests](https://github.com/BrainLesion/AURORA/actions/workflows/tests.yml/badge.svg)](https://github.com/BrainLesion/AURORA/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
<!-- [![codecov](https://codecov.io/gh/BrainLesion/BraTS/graph/badge.svg?token=A7FWUKO9Y4)](https://codecov.io/gh/BrainLesion/BraTS) -->

Deep learning models for brain cancer metastasis segmentation based on the manuscripts:
* [Identifying core MRI sequences for reliable automatic brain metastasis segmentation](https://www.medrxiv.org/content/10.1101/2023.05.02.23289342v1)
* [Development and external validation of an MRI-based neural network for brain metastasis segmentation in the AURORA multicenter study](https://www.sciencedirect.com/science/article/pii/S0167814022045625)

## Installation
With a Python 3.8+ environment, you can install `brainles_aurora` directly from [pypi.org](https://pypi.org/project/brainles-aurora/):

```
pip install brainles-aurora
```

## Recommended Environment

- CUDA 11.4+ (https://developer.nvidia.com/cuda-toolkit)
- Python 3.8+
- GPU with CUDA support and at least 6GB of VRAM

## Usage
BrainLes features Jupyter Notebook [tutorials](https://github.com/BrainLesion/tutorials/tree/main/AURORA) with usage instructions.

A minimal example could look like this:

```python
    from brainles_aurora.inferer import AuroraInferer, AuroraInfererConfig

    config = AuroraInfererConfig(
        tta=False, cuda_devices="4"
    )  # disable tta for faster inference in this showcase
    inferer = AuroraInferer(config=config)

    inferer.infer(
        t1="t1.nii.gz",
        t1c="t1c.nii.gz",
        t2="t2.nii.gz",
        fla="fla.nii.gz",
        segmentation_file="segmentation.nii.gz",
        whole_tumor_unbinarized_floats_file="whole_network.nii.gz",
        metastasis_unbinarized_floats_file="metastasis_network.nii.gz",
        log_file="aurora.log",
    )

```

## Citation
Please support our development by citing the following manuscripts:

[Identifying core MRI sequences for reliable automatic brain metastasis segmentation](https://www.sciencedirect.com/science/article/pii/S016781402389795X)

```
@article{buchner2023identifying,
  title={Identifying core MRI sequences for reliable automatic brain metastasis segmentation},
  author={Buchner, Josef A and Peeken, Jan C and Etzel, Lucas and Ezhov, Ivan and Mayinger, Michael and Christ, Sebastian M and Brunner, Thomas B and Wittig, Andrea and Menze, Bjoern H and Zimmer, Claus and others},
  journal={Radiotherapy and Oncology},
  volume={188},
  pages={109901},
  year={2023},
  publisher={Elsevier}
}
```

also consider citing the original AURORA manuscript, especially when using the `vanilla` model (all 4 modalities as input):

[Development and external validation of an MRI-based neural network for brain metastasis segmentation in the AURORA multicenter study](https://www.sciencedirect.com/science/article/pii/S0167814022045625)<>

```
@article{buchner2022development,
  title={Development and external validation of an MRI-based neural network for brain metastasis segmentation in the AURORA multicenter study},
  author={Buchner, Josef A and Kofler, Florian and Etzel, Lucas and Mayinger, Michael and Christ, Sebastian M and Brunner, Thomas B and Wittig, Andrea and Menze, Bj{\"o}rn and Zimmer, Claus and Meyer, Bernhard and others},
  journal={Radiotherapy and Oncology},
  year={2022},
  publisher={Elsevier}
}
```

## Contact / Feedback / Questions

If possible please open a GitHub issue [here](https://github.com/BrainLesion/AURORA/issues).

For inquiries not suitable for GitHub issues:

Florian Kofler
florian.kofler [at] tum.de

Josef Buchner
j.buchner [at] tum.de
