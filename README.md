[![PyPI version AURORA](https://badge.fury.io/py/brainles-aurora.svg)](https://pypi.python.org/pypi/brainles-aurora/)
[![Documentation Status](https://readthedocs.org/projects/brainles-aurora/badge/?version=latest)](http://brainles-aurora.readthedocs.io/?badge=latest)
[![tests](https://github.com/BrainLesion/AURORA/actions/workflows/tests.yml/badge.svg)](https://github.com/BrainLesion/AURORA/actions/workflows/tests.yml)

# AURORA
Deep learning models for brain cancer metastasis segmentation based on the manuscripts:
* [Identifying core MRI sequences for reliable automatic brain metastasis segmentation](https://www.medrxiv.org/content/10.1101/2023.05.02.23289342v1)
* [Development and external validation of an MRI-based neural network for brain metastasis segmentation in the AURORA multicenter study](https://www.sciencedirect.com/science/article/pii/S0167814022045625)

## Installation
With a Python 3.10+ environment, you can install directly from [pypi.org](https://pypi.org/project/brainles-aurora/):

```
pip install brainles-aurora
```

## Recommended Environment

- CUDA 11.4+ (https://developer.nvidia.com/cuda-toolkit)
- Python 3.8+
- GPU with CUDA support and at least 6GB of VRAM

## Usage
BrainLes features Jupyter Notebook [tutorials](https://github.com/BrainLesion/tutorials/tree/main/AURORA) with usage instructions.

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

also consider citing the original AURORA manuscript, especially when using the `vanilla` model:

[Development and external validation of an MRI-based neural network for brain metastasis segmentation in the AURORA multicenter study](https://www.sciencedirect.com/science/article/pii/S0167814022045625)

```
@article{buchner2022development,
  title={Development and external validation of an MRI-based neural network for brain metastasis segmentation in the AURORA multicenter study},
  author={Buchner, Josef A and Kofler, Florian and Etzel, Lucas and Mayinger, Michael and Christ, Sebastian M and Brunner, Thomas B and Wittig, Andrea and Menze, Bj{\"o}rn and Zimmer, Claus and Meyer, Bernhard and others},
  journal={Radiotherapy and Oncology},
  year={2022},
  publisher={Elsevier}
}
```


## Licensing

This project is licensed under the terms of the [GNU Affero General Public License v3.0](https://www.gnu.org/licenses/agpl-3.0.de.html).

Contact us regarding licensing.

## Contact / Feedback / Questions

If possible please open a GitHub issue [here](https://github.com/BrainLesion/AURORA/issues).

For inquiries not suitable for GitHub issues:

Florian Kofler
florian.kofler [at] tum.de

Josef Buchner
j.buchner [at] tum.de
