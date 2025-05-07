"""
Configuration utilities for loading model metadata.
"""
import os
import yaml
from typing import Dict, Any
from functools import lru_cache

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "config.yaml")

@lru_cache()
def get_config() -> Dict[str, Any]:
    """
    Load and parse the configuration file.
    
    Returns:
        Dict with the parsed configuration.
    """
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")
    
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)
    
    return config 