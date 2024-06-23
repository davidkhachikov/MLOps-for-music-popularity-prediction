"""
import of tested functions and types for tests
"""
from hydra import initialize, compose
from omegaconf import DictConfig
from src.data import (sample_data, sample_data_remotely)


data = sample_data()
def test_is_not_empty():
    """test of sampled data fails if it is empty"""
    assert not data.is_empty()

def test_size_of_sample():
    """test of sampled data fails if number of rows in sample does not correspond with configs"""
    # Initialize Hydra to read the configuration
    with initialize(config_path="../configs", version_base=None):
        cfg: DictConfig = compose(config_name='main')
    assert data.shape[0] == int(cfg.data_local.data_size*cfg.data_local.sample_size)

def test_number_of_columns():
    """test of sampled data fails if number of columns in sample does not correspond with configs"""
    # Initialize Hydra to read the configuration
    with initialize(config_path="../configs", version_base=None):
        cfg: DictConfig = compose(config_name='main')
    assert data.shape[1] == cfg.data_local.column_number
