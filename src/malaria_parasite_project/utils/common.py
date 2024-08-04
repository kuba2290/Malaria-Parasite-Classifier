import os
from box.exceptions import BoxValueError
import yaml
from src.logger import logging
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Read a YAML file and return its contents.
    
    Args: 
        path_to_yaml (Path): Path to the YAML file
    
    Raises:
        ValueError: If the YAML file is empty
        FileNotFoundError: If the file doesn't exist
        yaml.YAMLError: If there's an error in YAML parsing
    
    Returns:
        ConfigBox: ConfigBox type containing the YAML contents
    """
    try:
        with open(path_to_yaml, 'r') as yaml_file:
            content = yaml.safe_load(yaml_file)
            if not content:
                raise ValueError("YAML file is empty")
            logging.info(f"YAML file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except FileNotFoundError:
        logging.error(f"YAML file not found: {path_to_yaml}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {path_to_yaml}")
        raise
    except BoxValueError:
        logging.error(f"Error converting YAML content to ConfigBox: {path_to_yaml}")
        raise ValueError("YAML file is empty or invalid")
    except Exception as e:
        logging.error(f"Unexpected error reading YAML file: {path_to_yaml}")
        raise

@ensure_annotations
def create_directories(path_to_directories: list, verbose: bool = True):
    """
    Create a list of directories.
    
    Args:
        path_to_directories (list): List of paths of directories to create
        verbose (bool, optional): Log directory creation. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logging.info(f"Created directory at: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    """
    Save data as a JSON file.
    
    Args:
        path (Path): Path to save the JSON file
        data (dict): Dictionary to be saved as JSON
    """
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        logging.info(f"JSON file saved successfully at: {path}")
    except Exception as e:
        logging.error(f"Error saving JSON file at {path}: {str(e)}")
        raise

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Load data from a JSON file.

    Args:
        path (Path): Path to the JSON file

    Returns:
        ConfigBox: Data as class attributes instead of dict

    Raises:
        FileNotFoundError: If the JSON file is not found
        json.JSONDecodeError: If there's an error decoding the JSON
    """
    try:
        with open(path, 'r') as f:
            content = json.load(f)
        logging.info(f"JSON file loaded successfully from: {path}")
        return ConfigBox(content)
    except FileNotFoundError:
        logging.error(f"JSON file not found: {path}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from file {path}: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error loading JSON file {path}: {str(e)}")
        raise

@ensure_annotations
def save_bin(data: Any, path: Path):
    """
    Save data as a binary file.

    Args:
        data (Any): Data to be saved as binary
        path (Path): Path to save the binary file
    """
    try:
        joblib.dump(value=data, filename=path)
        logging.info(f"Binary file saved at: {path}")
    except Exception as e:
        logging.error(f"Error saving binary file at {path}: {str(e)}")
        raise

@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    Load data from a binary file.

    Args:
        path (Path): Path to the binary file

    Returns:
        Any: Object stored in the file

    Raises:
        FileNotFoundError: If the binary file is not found
    """
    try:
        data = joblib.load(path)
        logging.info(f"Binary file loaded from: {path}")
        return data
    except FileNotFoundError:
        logging.error(f"Binary file not found: {path}")
        raise
    except Exception as e:
        logging.error(f"Error loading binary file from {path}: {str(e)}")
        raise

@ensure_annotations
def get_size(path: Path) -> str:
    """
    Get the size of a file in KB.

    Args:
        path (Path): Path of the file

    Returns:
        str: Size in KB

    Raises:
        FileNotFoundError: If the file is not found
    """
    try:
        size_in_kb = round(os.path.getsize(path) / 1024)
        return f"~ {size_in_kb} KB"
    except FileNotFoundError:
        logging.error(f"File not found: {path}")
        raise
    except Exception as e:
        logging.error(f"Error getting file size for {path}: {str(e)}")
        raise

@ensure_annotations
def decode_image(imgstring: str, file_name: str):
    """
    Decode a base64 string and save it as an image file.

    Args:
        imgstring (str): Base64 encoded string of the image
        file_name (str): Name of the file to save the decoded image
    """
    try:
        imgdata = base64.b64decode(imgstring)
        with open(file_name, 'wb') as f:
            f.write(imgdata)
        logging.info(f"Image decoded and saved as: {file_name}")
    except Exception as e:
        logging.error(f"Error decoding image: {str(e)}")
        raise

@ensure_annotations
def encode_image_to_base64(image_path: Path) -> str:
    """
    Encode an image file to a base64 string.

    Args:
        image_path (Path): Path to the image file

    Returns:
        str: Base64 encoded string of the image

    Raises:
        FileNotFoundError: If the image file is not found
    """
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except FileNotFoundError:
        logging.error(f"Image file not found: {image_path}")
        raise
    except Exception as e:
        logging.error(f"Error encoding image to base64: {str(e)}")
        raise