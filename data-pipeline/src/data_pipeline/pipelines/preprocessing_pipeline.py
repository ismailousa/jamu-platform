from data_pipeline.utils.common import load_config
from data_pipeline.utils import logger
from data_pipeline.constants import CONFIG_FILE_PATH, ARTIFACTS_DIR
from data_pipeline.ingestion.load_metadata import load_metadatas
from data_pipeline.preprocessing.features import engineer_features

import pandas as pd
processed_dir = ARTIFACTS_DIR / "processed"

def run_preprocessing_pipeline(metadata_and_datasets):
    """
    Run the preprocessing pipeline.

    """
    metadatas = [m["metadata"] for m in metadata_and_datasets]
    names = [m["database_name"] for m in metadata_and_datasets]

    logger.info("Starting feature engineering...")
    features_and_metrics = engineer_features(metadatas)
    logger.info("Finished feature engineering.")
    
    for f, name in zip(features_and_metrics, names):
        database_dir = processed_dir / name
        metrics_dir = database_dir / "metrics"

        metadata = f["metadata"]
        age_stats = f["age_stats"]
        age_group_distribution = f["age_group_distribution"]
        gender_distribution = f["gender_distribution"]
        disease_distribution = f["disease_distribution"]
        disease_distribution_via_labels = f["disease_distribution_via_labels"]
        disease_correlation = f["disease_correlation"]
        distribution_metrics = f["distribution_metrics"]
        outliers_iqr_age = f["outliers_iqr_age"]

        database_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        metadata.to_csv(database_dir / "metadata.csv", index=False)
        age_stats.to_csv(metrics_dir / "age_stats.csv", index=False)
        age_group_distribution.to_csv(metrics_dir / "age_group_distribution.csv", index=False)
        gender_distribution.to_csv(metrics_dir / "gender_distribution.csv", index=False)
        disease_distribution.to_csv(metrics_dir / "disease_distribution.csv", index=False)
        disease_distribution_via_labels.to_csv(metrics_dir / "disease_distribution_via_labels.csv", index=False)
        disease_correlation.to_csv(metrics_dir / "disease_correlation.csv", index=False)
        distribution_metrics.to_csv(metrics_dir / "distribution_metrics.csv", index=False)
        
        outliers_iqr_age["outliers"].to_csv(metrics_dir / "outliers_iqr_age.csv", index=False)
        with open(metrics_dir / "outliers_iqr_age_bounds.txt", "w") as f:
            f.write(f"lower_bound: {outliers_iqr_age['lower_bound']}\n")
            f.write(f"upper_bound: {outliers_iqr_age['upper_bound']}\n")

        print(metadata.head())
        print(metadata.columns.tolist())
        
        print("\nAge Stats:")
        print(age_stats)
        print("\nAge Group Distribution:")
        print(age_group_distribution)
        print("\nGender Distribution:")
        print(gender_distribution)
        print("\nDisease Distribution:")
        print(disease_distribution)
        print("\nDisease Distribution via Labels:")
        print(disease_distribution_via_labels)
        print("\nDisease Correlation:")
        print(disease_correlation)
        print("\nDistribution Metrics:")
        print(distribution_metrics)
        print("\nOutliers IQR Age:")
        print(outliers_iqr_age)

    logger.info("Plotting visualisations...")
    # TODO: Plot visualisations
    logger.info("TODO: Plot visualisations")

    logger.info("Preprocssing images...")


    logger.info("Finished data preprocessing.")


if __name__ == "__main__":
    config = load_config(CONFIG_FILE_PATH)
    datasets = config.datasets
    metadatas_and_validations = load_metadatas(datasets, validate=True, clean=False)
    metadatas = [{"database_name": dataset["name"], "metadata": meta_and_val["metadata"]} 
             for meta_and_val, dataset in zip(metadatas_and_validations, datasets)]

    run_preprocessing_pipeline(metadatas)