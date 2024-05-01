from __future__ import annotations

import logging
import shutil
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, List

import requests
from brainles_aurora.inferer.constants import WEIGHTS_DIR_PATTERN

logger = logging.getLogger(__name__)

ZENODO_RECORD_URL = "https://zenodo.org/api/records/10557069"


def check_model_weights() -> Path:
    """Check if latest model weights are present and download them otherwise.

    Returns:
        Path: Path to the model weights folder.
    """
    package_folder = Path(__file__).parent.parent

    zenodo_metadata = _get_zenodo_metadata()

    matching_folders = list(package_folder.glob(WEIGHTS_DIR_PATTERN))
    # Get the latest downloaded weights
    latest_downloaded_weights = _get_latest_version_folder_name(matching_folders)

    if not latest_downloaded_weights:
        if not zenodo_metadata:
            logger.error(
                "Model weights not found locally and Zenodo could not be reached. Exiting..."
            )
            sys.exit()
        logger.info(
            f"Model weights not found. Downloading the latest model weights {zenodo_metadata['version']} from Zenodo..."
        )

        return _download_model_weights(
            package_folder=package_folder, zenodo_metadata=zenodo_metadata
        )

    logger.info(f"Found downloaded local weights: {latest_downloaded_weights}")

    if not zenodo_metadata:
        logger.warning(
            "Zenodo server could not be reached. Using the latest downloaded weights."
        )
        return package_folder / latest_downloaded_weights

    # Compare the latest downloaded weights with the latest Zenodo version
    if zenodo_metadata["version"] == latest_downloaded_weights.split("_v")[1]:
        logger.info(
            f"Latest model weights ({latest_downloaded_weights}) are already present."
        )
        return package_folder / latest_downloaded_weights

    logger.info(
        f"New model weights available on Zenodo ({zenodo_metadata['version']}). Deleting old and fetching new weights..."
    )
    # delete old weights
    shutil.rmtree(
        package_folder / latest_downloaded_weights,
        onerror=lambda func, path, excinfo: logger.warning(
            f"Failed to delete {path}: {excinfo}"
        ),
    )
    return _download_model_weights(
        package_folder=package_folder, zenodo_metadata=zenodo_metadata
    )


def _get_latest_version_folder_name(folders: List[Path]) -> str | None:
    """Get the latest (non empty) version folder name from the list of folders.

    Args:
        folders (List[Path]): List of folders matching the pattern.

    Returns:
        str | None: Latest version folder name if one exists, else None.
    """
    if not folders:
        return None
    latest_downloaded_folder = sorted(
        folders,
        reverse=True,
        key=lambda x: tuple(map(int, str(x).split("_v")[1].split("."))),
    )[0]
    # check folder is not empty
    if not list(latest_downloaded_folder.glob("*")):
        return None
    return latest_downloaded_folder.name


def _get_zenodo_metadata() -> Dict | None:
    """Get the metadata for the Zenodo record.

    Returns:
        dict: Metadata for the Zenodo record.
    """
    try:
        response = requests.get(ZENODO_RECORD_URL)
        return response.json()["metadata"]
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to fetch Zenodo metadata: {e}")
        return None


def _download_model_weights(package_folder: Path, zenodo_metadata: Dict) -> Path:
    """Download the latest model weights from Zenodo and extract them to the target folder.

    Args:
        package_folder (Path): Package folder path in which the model weights will be stored.
        zenodo_metadata (Dict): Metadata for the Zenodo record.

    Returns:
        Path: Path to the model weights folder.
    """
    weights_folder = package_folder / f"weights_v{zenodo_metadata['version']}"

    # ensure folder exists
    weights_folder.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Downloading model weights from Zenodo ({ZENODO_RECORD_URL}). This might take a while..."
    )
    # Make a GET request to the URL
    response = requests.get(f"{ZENODO_RECORD_URL}/files-archive")
    # Ensure the request was successful
    if response.status_code != 200:
        logger.error(
            f"Failed to download model weights. Status code: {response.status_code}"
        )
        return
    # Extract the downloaded zip file to the target folder
    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(weights_folder)
    logger.info(f"Zip file extracted successfully to {weights_folder}")
    return weights_folder
