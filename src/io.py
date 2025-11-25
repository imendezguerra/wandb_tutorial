"""Functions to save and load data."""

from pathlib import Path
from typing import Any, Dict, Union
import yaml


def yaml_load(
    file_path: Union[str, Path],
) -> Dict[str, Any]:
    """Load a YAML file into a dictionary.

    Args:
        file_path (Union[str, Path]): Path to the YAML file.

    Returns:
        Dict[str, Any]: Dictionary containing the YAML data.

    """
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    return data


def yaml_save(
    data: Dict[str, Any],
    file_path: Union[str, Path],
) -> None:
    """Save a dictionary to a YAML file.
    
    Args:
        data (Dict[str, Any]): Data to save.
        file_path (Union[str, Path]): Path to the YAML file.

    """
    with open(file_path,'w') as f:
        yaml.dump(data, f)
