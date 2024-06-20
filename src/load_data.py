"""
Module for loading data related to music popularity prediction.

This module contains functions for reading and processing data from CSV files,
using DVC for version control and fetching specific versions of the data.
"""

import os.path
import pandas as pd
import hydra
import dvc.api
import gdown

@hydra.main(config_path='../configs', config_name='main', version_base=None)
def read_data_with_yaml(cfg=None):
    """
    Reads data from a CSV file located in the data directory using DVC to fetch the 
    version specified in main.yaml.

    Parameters:
    - cfg (dict): Configuration dictionary containing details about the data source.

    Returns:
    - DataFrame: A pandas DataFrame containing the data from the CSV file.
    """
    url = dvc.api.get_url(
        path=os.path.join(cfg.data.path),
        repo=os.path.join(cfg.data.repo),
        rev=cfg.data.version,
        remote=cfg.data.remote
    )
    return pd.read_csv(url)

df = read_data_with_yaml()
print(df.head())
