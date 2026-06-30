# Path is the main class used to represent and manipulate file/directory paths.
from pathlib import Path 

# It allows you to access dictionary keys as attributes (dot notation), making configs easier to handle.
from box import ConfigBox  

# Useful for configuration files, since YAML is human-readable and widely used for settings.
import yaml  

# It enforces type annotations at runtime, ensuring that function inputs/outputs match the declared types.
from ensure import ensure_annotations  
from cnnClassifier.logger import logger 


@ensure_annotations
def read_yaml(path_to_yaml:Path)->ConfigBox:
    """
    Reads a YAML file and returns its contents as a ConfigBox.
    """
    with open(path_to_yaml) as yaml_file:
        content=yaml.safe_load(yaml_file)

        logger.info(f"YAML file loaded successfully: {path_to_yaml}")

        return ConfigBox(content)