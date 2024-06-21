"""
Module for loading data related to music popularity prediction.

This module contains functions for reading and processing data from CSV files,
using DVC for version control and fetching specific versions of the data.
"""
import math
import os.path
import pandas as pd
from hydra import initialize, compose
import hydra
from omegaconf import DictConfig
import dvc.api


def sample_data():
    """
    Loads a sample of music popularity data from a CSV file using DVC for version control.

    This function initializes Hydra to read configurations, then uses DVC to open and read 
    a specific version of the data file.

    Returns:
        df: sampled 
    """
    # Initialize Hydra to read the configuration
    with initialize(config_path="../configs", version_base=None):
        cfg: DictConfig = compose(config_name='main')

    # Read the data file
    with dvc.api.open(cfg.data_local.path, rev=cfg.data_local.version, encoding='utf-8') as f:
        df = pd.read_csv(f)
    return df


def sample_data_remotely():
    """
    Reads data from a CSV file located in the data directory using DVC to fetch the 
    version specified in main.yaml.

    Parameters:
    - cfg (dict): Configuration dictionary containing details about the data source.

    Returns:
    - DataFrame: A pandas DataFrame containing the data from the CSV file.
    """
    with initialize(config_path="../configs", version_base=None):
        cfg: DictConfig = compose(config_name='main')
    url = dvc.api.get_url(
        path=os.path.join(cfg.data_remote.path),
        repo=os.path.join(cfg.data_remote.repo),
        rev=cfg.data_remote.version,
        remote=cfg.data_remote.remote
    )
    df = pd.read_csv(url)
    return df.sample(math.ceil(len(df) * cfg.data_remote.sample_size), random_state=42)


if __name__ == "__main__":
    sample_data()
