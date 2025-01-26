import os
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
from data_pipeline.utils.common import load_config
from data_pipeline.utils import logger
from pathlib import Path

from data_pipeline.constants import CONFIG_FILE_PATH
from data_pipeline.constants import ARTIFACTS_DIR



def download_odir_dataset(kaggle_dataset, download_dir):
    """
    Downloads the ODIR dataset from Kaggle.

    Parameters:
    - kaggle_dataset: The Kaggle dataset to download.
    - download_dir: The directory in which to save the downloaded dataset.

    """
    try:
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(kaggle_dataset, path=download_dir, unzip=True)
        logger.info(f"Saved dataset to {download_dir}")
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")

def download_datasets(datasets):
    """
    Downloads the datasets specified in the config file.

    Parameters:
    - datasets: The datasets to download.

    """
    downloaded_datasets = []
    for dataset in datasets:
        if dataset.source == "kaggle":
            dataset_name = dataset.name
            dataset_id = dataset.kaggle_dataset
            download_dir = Path(dataset.get("download_dir", ARTIFACTS_DIR / "raw/")) / dataset_name
            download_dir.mkdir(parents=True, exist_ok=True)
            download_odir_dataset(dataset_id, download_dir)
            downloaded_datasets.append(dataset)

    return downloaded_datasets


if __name__ == "__main__":
    config = load_config(CONFIG_FILE_PATH)
    datasets = config.datasets
    download_datasets(datasets)