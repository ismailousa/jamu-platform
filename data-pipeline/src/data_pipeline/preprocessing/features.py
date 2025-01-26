import pandas as pd
import numpy as np
import ast

def parse_target_column(metadata: pd.DataFrame, target_column: str):
    """
    Parse the target column from the metadata.
    """
    def parse_target(x):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []

    metadata['parsed_target'] = metadata[target_column].apply(parse_target)
    return metadata

def parse_gender_column(metadata: pd.DataFrame, gender_column: str):
    """One-hot encodes"""
    metadata["parsed_gender"] = metadata[gender_column].map({'Male': 0, 'Female': 1})
    return metadata

def get_age_distribution(metadata: pd.DataFrame, age_col: str = "Patient Age") -> pd.DataFrame:
    if age_col not in metadata.columns:
        raise ValueError(f"Column '{age_col}' not found in metadata")
    age_stats = metadata[age_col].describe().to_frame().T
    return age_stats

def get_age_group_distribution(metadata: pd.DataFrame, age_col: str = "Patient Age", bin_size: int = 10) -> pd.DataFrame:
    if age_col not in metadata.columns:
        raise ValueError(f"Column '{age_col}' not found in metadata")

    bins = range(0, metadata[age_col].max() + bin_size, bin_size)
    labels = [f'{i}-{i+bin_size-1}' for i in range(0, metadata[age_col].max(), bin_size)]

    metadata['age_group'] = pd.cut(metadata[age_col], bins=bins, labels=labels)
    age_distribution = metadata['age_group'].value_counts().sort_index().reset_index()
    age_distribution.columns = ['age_group', 'count']
    return age_distribution

def get_gender_distribution(metadata: pd.DataFrame, gender_col: str = "Patient Sex") -> pd.DataFrame:
    gender_counts = metadata[gender_col].value_counts().reset_index()
    gender_counts.columns = ['Gender', 'Count']

    gender_counts['Proportion'] = gender_counts['Count'] / gender_counts['Count'].sum()
    return gender_counts

def get_disease_distribution(metadata: pd.DataFrame, target_col: str = "parsed_target", labels_mapper: list[str] = None) -> pd.DataFrame:
    target_values = np.array(metadata[target_col].tolist())
    
    disease_counts = target_values.sum(axis=0)
    
    if labels_mapper is None:
        labels_mapper = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    
    disease_distribution = pd.DataFrame({
        'Disease': labels_mapper,
        'Count': disease_counts
    })
    
    return disease_distribution

def get_disease_distribution_via_labels(metadata: pd.DataFrame, target_col: str = "labels", labels_mapper: list[str] = None) -> pd.DataFrame:
    disease_counts = metadata[target_col].value_counts().reset_index()
    disease_counts.columns = ['Disease', 'Count']
    return disease_counts

def get_disease_correlation(metadata: pd.DataFrame) -> pd.DataFrame:

    numeric_df = metadata.select_dtypes(include=[np.number])
    disease_correlation = numeric_df.corr()
    
    return disease_correlation

def get_distribution_metrics(metadata: pd.DataFrame) -> pd.DataFrame:
    metrics = pd.DataFrame(columns=['Column', 'Skewness', 'Kurtosis'])
    numeric_df = metadata.select_dtypes(include=[np.number])

    for col in numeric_df.columns:
        skewness = numeric_df[col].skew()
        kurtosis = numeric_df[col].kurtosis()
        metrics = pd.concat([metrics, pd.DataFrame({'Column': [col], 'Skewness': [skewness], 'Kurtosis': [kurtosis]})], ignore_index=True)

    return metrics

def detect_outliers_iqr(df: pd.DataFrame, 
                        column: str, 
                        multiplier: float = 1.5) -> pd.DataFrame:
    """
    Detects outliers in a specified column using the IQR method.
    
    Args:
        df (pd.DataFrame): The metadata dataframe.
        column (str): The column to check for outliers.
        multiplier (float): The IQR multiplier to define outlier thresholds.
    
    Returns:
        pd.DataFrame: DataFrame containing the outlier data points.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    print(Q1, Q3, IQR)
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    print(lower_bound, upper_bound)
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    result = {
        "outliers": outliers,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound
    }
    return result


def engineer_features(metadatas: list[pd.DataFrame]):
    """
    Engineer features from the metadata.
    """
    features_and_metrics = []
    for metadata in metadatas:
        metadata = parse_target_column(metadata, 'target')
        metadata = parse_gender_column(metadata, 'Patient Sex')

        age_stats = get_age_distribution(metadata)
        age_group_distribution = get_age_group_distribution(metadata)
        gender_distribution = get_gender_distribution(metadata)
        disease_distribution = get_disease_distribution(metadata)
        disease_distribution_via_labels = get_disease_distribution_via_labels(metadata)
        disease_correlation = get_disease_correlation(metadata)
        distribution_metrics = get_distribution_metrics(metadata)
        outliers_iqr_age = detect_outliers_iqr(metadata, 'Patient Age')

        features_and_metrics.append({
            "metadata": metadata,
            "age_stats": age_stats,
            "age_group_distribution": age_group_distribution,
            "gender_distribution": gender_distribution,
            "disease_distribution": disease_distribution,
            "disease_distribution_via_labels": disease_distribution_via_labels,
            "disease_correlation": disease_correlation,
            "distribution_metrics": distribution_metrics,
            "outliers_iqr_age": outliers_iqr_age,
        })

        

    return features_and_metrics