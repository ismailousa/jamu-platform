import pandas as pd
from data_pipeline.utils import logger
from data_pipeline.constants import ARTIFACTS_DIR

def load_metadata(dataset_name: str, metadata_file: str, raw: bool = True):
    """
    Load metadata from a file."""
    try:
        raw_processed = "raw" if raw else "processed"
        metadata_path = ARTIFACTS_DIR / raw_processed / dataset_name / metadata_file
        metadata = pd.read_csv(metadata_path)
        logger.info(f"Loaded metadata for {dataset_name}")
        return metadata
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")


def save_metadata(metadata: pd.DataFrame, dataset_name: str, metadata_file: str, raw: bool = True):
    """
    Save metadata to a file."""
    try:
        raw_processed = "raw" if raw else "processed"
        metadata_path = ARTIFACTS_DIR / raw_processed / dataset_name / metadata_file
        metadata.to_csv(metadata_path, index=False)
        logger.info(f"Saved metadata for {dataset_name}")
    except Exception as e:
        logger.error(f"Error saving metadata: {e}")

def check_null_or_empty(metadata):
    """
    Check for null or empty values in the metadata."""
    null_or_empty = metadata.isnull() | (metadata == "")
    rows_with_null_or_empty = null_or_empty.any(axis=1)

    if rows_with_null_or_empty.any():
        for idx in metadata[rows_with_null_or_empty].index:
            missing_columns = metadata.columns[rows_with_null_or_empty.loc[idx]].tolist()
            logger.warning(f"Row {idx} has missing values in columns: {missing_columns}")
    logger.info(f"Found {rows_with_null_or_empty.sum()} rows with null or empty values in the metadata")
    return rows_with_null_or_empty.sum()

def remove_null_or_empty(metadata):
    """
    Remove rows with null or empty values from the metadata."""
    metadata = metadata.dropna()
    metadata = metadata.dropna(subset=metadata.columns[metadata.dtypes == "object"])
    return metadata

def check_duplicates(metadata):
    """
    Check for duplicate rows in the metadata."""
    duplicates = metadata.duplicated('filename', keep=False)
    if duplicates.any():
        logger.warning(f"Found {duplicates.sum()} duplicate rows in the metadata")
    return duplicates.sum()

def remove_duplicates(metadata):
    """
    Remove duplicate rows from the metadata."""
    metadata = metadata.drop_duplicates('filename')
    return metadata

def check_files_exist(metadata, dataset_name: str, image_folder: str, raw: bool = True):
    """
    Check if the files in the metadata exist in the raw or processed directory."""
    raw_processed = "raw" if raw else "processed"
    files_exist = metadata['filename'].apply(lambda x: (ARTIFACTS_DIR / raw_processed / dataset_name / image_folder / x).exists())
    missing_files = []
    if not files_exist.all():
        missing_files = metadata[~files_exist]['filename'].tolist()
        logger.warning(f"Missing files in {raw_processed} directory: {missing_files}")
    return len(missing_files)

def remove_missing_files(metadata, dataset_name: str, image_folder: str, raw: bool = True):
    """
    Remove rows with missing files from the metadata."""
    raw_processed = "raw" if raw else "processed"
    files_exist = metadata['filename'].apply(lambda x: (ARTIFACTS_DIR / raw_processed / dataset_name / image_folder / x).exists())
    metadata = metadata[files_exist]
    return metadata

def check_left_right_pairs(metadata):
    """
    Check for left-right pairs in the metadata."""
    metadata = metadata.copy()
    metadata['number'] = metadata['filename'].apply(lambda x: x.split("_")[0])
    metadata['eye'] = metadata['filename'].apply(lambda x: x.split("_")[1].split(".")[0])   

    pivot_table = metadata.pivot(index='number', columns='eye', values='filename', aggfunc='count', fill_value=0)
    pairs = pivot_table[(pivot_table['left'] == 1) & (pivot_table['right'] == 1)]
    not_pairs = pivot_table[(pivot_table['left'] == 0) | (pivot_table['right'] == 0)]
    too_many = pivot_table[(pivot_table['left'] > 1) | (pivot_table['right'] > 1)]

    return pairs, not_pairs, too_many

def clean_metadata(metadata, dataset_name: str, raw: bool = True):
    """
    Clean the metadata by removing rows with missing values, duplicates, and missing files."""
    metadata = remove_null_or_empty(metadata)
    metadata = remove_duplicates(metadata)
    metadata = remove_missing_files(metadata, dataset_name, raw)
    return metadata

def validate_metadata(metadata, dataset_name: str, image_folder: str, raw: bool = True):
    """
    Validate the metadata by checking for missing values, duplicates, and missing files."""
    null_or_empty = check_null_or_empty(metadata)
    duplicates = check_duplicates(metadata)
    missing_files = check_files_exist(metadata, dataset_name, image_folder, raw)
    return null_or_empty, duplicates, missing_files

def load_metadatas(datasets, validate=False, clean=False):
    """
    Load metadata for multiple datasets."""
    metadatas = []
    for dataset in datasets:
        dataset_name = dataset.name
        metadata_file = dataset.metadata_file
        metadata = load_metadata(dataset_name, metadata_file)


        null_or_empty, duplicates, missing_files = None, None, None
        image_folder = dataset.get("image_folder", "images")

        if validate:
            null_or_empty, duplicates, missing_files = validate_metadata(metadata, dataset_name, image_folder)
            logger.info(f"Validation results for {dataset_name}: null_or_empty={null_or_empty}, duplicates={duplicates}, missing_files={missing_files}")

        if clean:
            metadata = clean_metadata(metadata, dataset_name)
            # save_metadata(metadata, dataset_name, metadata_file)

        result = {
            "dataset_name": dataset_name,
            "null_or_empty": null_or_empty,
            "duplicates": duplicates,
            "missing_files": missing_files,
            "metadata": metadata
        }
        metadatas.append(result)
    return metadatas

