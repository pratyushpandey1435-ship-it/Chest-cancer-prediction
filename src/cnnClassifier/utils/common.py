# Utility
# functions that will  be used again and again will be stored here and will be called when needed
import os
from box.exceptions import BoxValueError #alternate of the exceptions.py
import yaml
from cnnClassifier import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input
    
    Raises:
        ValueError: if yaml file  is empty
        e: empty file
    
    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(path_to_directories:list ,verbose=True):
    """ create list of directories

    Args:
        path_to_directories (list): list of the path to the directories
        ignore_log (bool, optional) : ignore if multiple dirs is to be created. Defaults to False 
    """
    for path in path_to_directories:
        os.makedirs(path,exist_ok=True)
    if verbose:
        logger.info(f"created directory at: {path}")

@ensure_annotations
def save_json(path:Path, data:dict):
    """ save json data

    Args:
        path (Path): path to the json file
        data (dict): data to be saved in json file
    """
    with open(path,"w") as f:
        json.dump(data,f,indent=4)

    logger.info(f"json file saved at: {path}")

@ensure_annotations
def load_json(path:Path) -> ConfigBox:
    """ load json files data

    Args:
        path (Path): apth to the json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """

    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded successfuly from : {path}")
    return ConfigBox(content)

@ensure_annotations
def save_bin(data:Any , path:Path):
    """save binary file
    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at: {path}")

@ensure_annotations
def load_bin(path:Path) -> Any:
    """load binary data
    Args:
    path (Path): path tbianry file
    
    Returns:
        Any objectstored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB
    
    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"- {size_in_kb} KB"


@ensure_annotations
def decodeImage(imgstring, filename):
    imgdata = base64.b64decode(imgstring)
    with open(filename, "wb") as f:
        f.write(imgdata)
        f.close()

@ensure_annotations
def encodeImage(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())
    