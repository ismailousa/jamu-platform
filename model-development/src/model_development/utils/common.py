import yaml
from pathlib import Path

from model_development.utils import logger
from box import ConfigBox

def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Read a YAML file and return a ConfigBox object.

    Parameters:
    - path_to_yaml: The path to the YAML file to read.

    Returns:
    - config: The ConfigBox object containing the YAML file data.

    """
    try:
        with open(path_to_yaml, "r") as file:
            content = yaml.safe_load(file)
            logger.info(f"Loaded yaml from {path_to_yaml}")
            return ConfigBox(content)
    except Exception as e:
        logger.error(f"Error reading yaml: {e}")
    

def load_config(config_path: Path) -> ConfigBox:
    """
    Load a ConfigBox object from a YAML file.

    Parameters:
    - config_path: The path to the YAML file to load.

    Returns:
    - config: The ConfigBox object containing the YAML file data.

    """
    return read_yaml(config_path)