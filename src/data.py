"""
Module for loading data related to music popularity prediction.

This module contains functions for reading and processing data from CSV files,
using DVC for version control and fetching specific versions of the data.
"""
import math
import os.path
import pandas as pd
from hydra import initialize, compose
from omegaconf import DictConfig
import dvc.api


def sample_data():
    """
    Loads a sample of music popularity data from a CSV file using DVC for version control.

    This function initializes Hydra to read configurations, then uses DVC to open and read 
    a specific version of the data file. It also attempts to save the sampled data to a local CSV file.

    Returns:
        None
    """
    try:
        with initialize(config_path="../configs", version_base=None):
            cfg: DictConfig = compose(config_name='main')

        with dvc.api.open(cfg.data_local.path, rev=cfg.data_local.version, encoding='utf-8') as f:
            df = pd.read_csv(f)

        sampled_df = df.iloc[0:int(len(df) * cfg.data_local.sample_size)]
        current_dir = os.getcwd() 
        parent_dir = os.path.dirname(current_dir)
        target_folder = os.path.join(parent_dir, 'data', 'samples')
        os.makedirs(os.path.dirname(target_folder), exist_ok=True)
        sampled_df.to_csv(target_folder, index=False)

        print(f"Sampled data saved to {target_folder}")

    except Exception as e:
        print(f"An error occurred while saving the sampled data: {e}")


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
    file = dvc.api.read(
        path=os.path.join(cfg.data_remote.path),
        repo=os.path.join(cfg.data_remote.repo),
        rev=cfg.data_remote.version,
        remote=cfg.data_remote.remote,
        encoding='utf-8'
    )
    # print('\n\n\n' + url + '\n\n\n')
    df = pd.read_csv(file)
    return df.sample(math.ceil(len(df) * cfg.data_remote.sample_size), random_state=42)


if __name__ == "__main__":
    sample_data_remotely()
