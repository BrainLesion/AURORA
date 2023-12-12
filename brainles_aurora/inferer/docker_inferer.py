import glob
import logging
import os
import os.path as op
import tempfile
from pathlib import Path

import docker
import yaml

from brainles_aurora.inferer.dataclasses import DockerInfererConfig
from brainles_aurora.inferer.inferer import AbstractInferer
from brainles_aurora.utils import own_itk as oitk


class DockerInferer(AbstractInferer):
    def __init__(self, config: DockerInfererConfig) -> None:
        super().__init__(config=config)
        # load docker config
        with open(
            "/Users/marcelrosier/Projects/helmholtz/AURORA/brainles_aurora/config/docker.yml",
            "r",
        ) as file:
            data = yaml.safe_load(file)
            self.docker_config = data[self.config.container_id]

        with open(
            "/Users/marcelrosier/Projects/helmholtz/AURORA/brainles_aurora/config/fileformats.yml",
            "r",
        ) as file:
            data = yaml.safe_load(file)
            self.ff = data[self.docker_config["fileformat"]]

        print(self.docker_config)
        self.inputs = {
            key: getattr(self.config, key)
            for key in ["t1", "t1c", "t2", "fla"]
            if getattr(self.config, key)
        }
        self.resultsDir = None

    def _handle_results(self, directory):
        outputPath = (
            self.output_folder / f"{self.config.container_id}_segmentation.nii.gz"
        )
        contents = glob.glob(
            op.join(directory, f"tumor_{self.config.container_id}_class.nii*")
        )
        if not contents:
            contents = glob.glob(op.join(directory, "tumor_*_class.nii*"))
        if not contents:
            contents = glob.glob(
                op.join(directory, f"{self.config.container_id}*.nii*")
            )
        if not contents:
            contents = glob.glob(op.join(directory, "*tumor*.nii*"))
        img = oitk.get_itk_image(contents[0])
        logging.info(f"Writing result: {outputPath}")
        oitk.write_itk_image(img, str(outputPath))

    def _setup(self, storage: str):
        self.tempDir = storage
        self.resultsDir = op.join(self.tempDir, "results")
        os.mkdir(self.resultsDir)
        os.chmod(self.resultsDir, 0o777)

        for key, img in self.inputs.items():
            savepath = op.join(self.tempDir, self.ff[key])
            logging.info(f"Writing to path {savepath}")
            if isinstance(img, (str, Path)):
                img = oitk.get_itk_image(img)
            oitk.write_itk_image(img, savepath)

    def _infer(self):
        # Create a Docker client
        client = docker.from_env()

        # Define the volume mapping
        volume_mapping = {
            self.tempDir: {"bind": self.docker_config["mountpoint"], "mode": "rw"}
        }
        logging.info(f"Volume Mapping: {volume_mapping}")

        # Run the container
        container = client.containers.run(
            self.docker_config["image"],
            remove=True,
            volumes=volume_mapping,
            command=self.docker_config["command"],
        )

        self._handle_results(self.resultsDir)

    def infer(self):
        self.package_directory = Path(
            os.path.dirname(os.path.abspath(__file__))
        ).parent.parent

        with tempfile.TemporaryDirectory(dir=self.package_directory) as storage:
            self._setup(storage=storage)
            self._infer()
