"""
import of tested functions and types for tests
"""
from hydra import initialize, compose
from omegaconf import DictConfig
import pytest
import pandas as pd

samples_data = [
    ("../data/samples/sample_1.csv"),
    ("../data/samples/sample_2.csv"),
    ("../data/samples/sample_3.csv"),
    ("../data/samples/sample_4.csv"),
    ("../data/samples/sample_5.csv")
]

data_list = [pd.read_csv(path) for path in samples_data]

@pytest.mark.parametrize("data", data_list)
def test_is_not_empty(data):
    """test of sampled data fails if it is empty"""
    assert not data.is_empty()

@pytest.mark.parametrize("data", data_list)
def test_size_of_sample(data):
    """test of sampled data fails if number of rows in sample does not correspond with configs"""
    # Initialize Hydra to read the configuration
    with initialize(config_path="../configs", version_base=None):
        cfg: DictConfig = compose(config_name='main')
    assert data.shape[0] == int(cfg.data.data_size*cfg.data.sample_size)

@pytest.mark.parametrize("data", data_list)
def test_number_of_columns(data):
    """test of sampled data fails if number of columns in sample does not correspond with configs"""
    # Initialize Hydra to read the configuration
    with initialize(config_path="../configs", version_base=None):
        cfg: DictConfig = compose(config_name='main')
    assert data.shape[1] == cfg.data.column_number
