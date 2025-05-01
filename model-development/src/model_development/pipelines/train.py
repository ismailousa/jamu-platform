from model_development.utils import logger
from model_development.constants import CONFIG_FILE
from model_development.utils.common import load_config
from model_development.core.datasets import ImageDataset
from model_development.preprocessing.torch_transforms import get_training_transforms_224, get_validation_transforms_224


def run_training_pipeline(args, upload=False):
    """
    Run the training pipeline.

    """
    logger.info("Starting training pipeline...")

    ## Process the data
    transform_train = get_training_transforms_224()
    transform_val = get_validation_transforms_224()

    full_dataset = ImageDataset(args.image_dir, notebook_dir / args.csv, transform=transform_train)

    # Split the dataset
    dataset_size = len(full_dataset)

    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # Set the transforms
    train_dataset.transform = transform_train
    val_dataset.transform = transform_val
    test_dataset.transform = transform_val


    # Compute weights for the training set to balance classes.
    train_labels = [full_dataset[idx][1] for idx in train_dataset.indices]
    class_counts = np.bincount(train_labels, minlength=args.num_classes)
    class_counts = np.where(class_counts == 0, 1, class_counts)  # Avoid division by zero.
    class_weights = 1.0 / class_counts
    samples_weight = [class_weights[label] for label in train_labels]
    samples_weight = torch.DoubleTensor(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)


    # Create DataLoaders.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    device = torch.device('mps' if torch.mps.is_available() else 'cpu')

    model = load_resnet18_torch(args.num_classes, weights="pretrained", mode="finetune", device=device)

    criterion, optimizer, scheduler = get_hyperparameters()

    writer = SummaryWriter(log_dir=args.log_dir)

    # Add training code here
    logger.info("Finished training pipeline.")

if __name__ == "__main__":
    config = load_config(CONFIG_FILE)
    print(config)
    run_training_pipeline(config)


# # from data_pipeline.utils.common import load_config
# # from model_development.utils import logger
# # from data_pipeline.constants import CONFIG_FILE_PATH
# # from data_pipeline.ingestion.fetch_kaggle import download_datasets
# # from data_pipeline.ingestion.load_metadata import load_metadatas

# def run_ingestion_pipeline(datasets, download=False, upload=False):
#     """
#     Run the ingestion pipeline.

#     """
#     # Download the datasets
#     logger.info("Starting data ingestion...")
#     datasets = download_datasets(datasets) # TOFIX:
#     datasets = [datasets[0]]

#     # Upload the datasets to S3 optionally
#     if any(dataset['upload_to_s3'] for dataset in datasets):
#         logger.info("Uploading datasets to S3...")
#         upload_datasets(datasets)

#     # Load and validate the metadatas
#     metadatas_and_validations = load_metadatas(datasets, validate=True, clean=False)
#     logger.info("Finished data ingestion.")

#     for m in metadatas_and_validations:
#         metadata = m["metadata"]
#         logger.info(f"Number of rows in metadata: {metadata.shape[0]}")
#         logger.info(f"Number of columns in metadata: {metadata.shape[1]}")
#         logger.info(f"Metadata columns: {metadata.columns.tolist()}")
#         logger.info(f"Metadata head: \n{metadata.head()}")

    

# # if __name__ == "__main__":
# #     config = load_config(CONFIG_FILE_PATH)
# #     datasets = config.datasets
# #     run_ingestion_pipeline(datasets)