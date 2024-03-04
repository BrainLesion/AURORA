from __future__ import annotations

import logging
import os
import tempfile
import zipfile
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

DOWNLOAD_URL = "https://zenodo.org/api/records/10557069/files-archive"


def download_model_weights(target_folder: str | Path):
    """Download the model weights from Zenodo and extract them to the target folder.

    Args:
        target_folder (str | Path): The folder to which the model weights should be downloaded and extracted to.
    """
    # Create the target folder if it does not exist
    os.makedirs(target_folder, exist_ok=True)

    # Create a temporary file to store the downloaded zip file
    with tempfile.NamedTemporaryFile() as temp_zip_file:
        temp_zip_file_path = temp_zip_file.name

        logger.info(
            f"Downloading model weights from Zenodo ({DOWNLOAD_URL}). This might take a while..."
        )
        # Make a GET request to the URL
        response = requests.get(DOWNLOAD_URL)

        # Ensure the request was successful
        if response.status_code != 200:
            logger.error(
                f"Failed to download model weights from {DOWNLOAD_URL}. Status code: {response.status_code}"
            )
            return

        # Save the zip file locally
        temp_zip_file.write(response.content)

        logger.info("Zip file downloaded successfully")

        # Extract the contents of the zip file
        with zipfile.ZipFile(temp_zip_file_path, "r") as zip_ref:
            zip_ref.extractall(target_folder)

        logger.info(f"Zip file extracted successfully to {target_folder}")
