from data_pipeline.utils.common import load_config
from data_pipeline.utils import logger
from data_pipeline.constants import CONFIG_FILE_PATH
from data_pipeline.ingestion.fetch_kaggle import download_datasets
from data_pipeline.ingestion.load_metadata import load_metadatas

def run_ingestion_pipeline(datasets):
    """
    Run the ingestion pipeline.

    """
    # Download the datasets
    logger.info("Starting data ingestion...")
    # datasets = download_datasets(datasets) # TOFIX:
    datasets = [datasets[0]]

    # Upload the datasets to S3 optionally
    if any(dataset['upload_to_s3'] for dataset in datasets):
        logger.info("Uploading datasets to S3...")
        upload_datasets(datasets)

    # Load and validate the metadatas
    metadatas_and_validations = load_metadatas(datasets, validate=True, clean=False)
    logger.info("Finished data ingestion.")

    for m in metadatas_and_validations:
        metadata = m["metadata"]
        logger.info(f"Number of rows in metadata: {metadata.shape[0]}")
        logger.info(f"Number of columns in metadata: {metadata.shape[1]}")
        logger.info(f"Metadata columns: {metadata.columns.tolist()}")
        logger.info(f"Metadata head: \n{metadata.head()}")

    

if __name__ == "__main__":
    config = load_config(CONFIG_FILE_PATH)
    datasets = config.datasets
    run_ingestion_pipeline(datasets)