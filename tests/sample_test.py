"""
import of tested functions and types for tests
"""
from hydra import initialize, compose
from omegaconf import DictConfig
import pytest
import pandas as pd
from math import ceil

samples_data = [
    ("./data/samples/sample.csv")
]

data_list = [pd.read_csv(path) for path in samples_data]

@pytest.mark.parametrize("data", data_list)
def test_is_not_empty(data):
    """test of sampled data fails if it is empty"""
    assert not data.empty

@pytest.mark.parametrize("data", data_list)
def test_size_of_sample(data):
    """test of sampled data fails if number of rows in sample does not correspond with configs"""
    # Initialize Hydra to read the configuration
    with initialize(config_path="../configs", version_base=None):
        cfg: DictConfig = compose(config_name='main')
    assert data.shape[0] == ceil(cfg.data.data_size/cfg.data.num_files)

@pytest.mark.parametrize("data", data_list)
def test_number_of_columns(data):
    """test of sampled data fails if number of columns in sample does not correspond with configs"""
    # Initialize Hydra to read the configuration
    with initialize(config_path="../configs", version_base=None):
        cfg: DictConfig = compose(config_name='main')
    assert data.shape[1] == cfg.data.column_number
