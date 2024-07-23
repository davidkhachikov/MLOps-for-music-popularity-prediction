# test_sample_data.py
import pytest
import pandas as pd
from src.data import sample_data
from src.utils import init_hydra
from math import ceil
import numpy as np
import os

sample_path = "./data/samples/sample.csv"
data_path = "./data/raw/tracks.csv"

data_shape = pd.read_csv(data_path, low_memory=False).shape
test_tuples = [(sample_path, data_shape)]
BASE_PATH = os.getenv('PROJECTPATH')

sample_data(BASE_PATH)

@pytest.mark.parametrize("sample_path, data_shape", test_tuples)
def test_size_of_sample(sample_path, data_shape):
    """test of sampled data fails if number of rows in sample does not correspond with configs"""
    # Initialize Hydra to read the configuration
    cfg = init_hydra()
    sample_len = pd.read_csv(sample_path).shape[0]
    data_len = data_shape[0]
    assert np.abs(sample_len - ceil(data_len/cfg.data.num_samples)) < cfg.data.num_samples

@pytest.mark.parametrize("sample_path, data_shape", test_tuples)
def test_number_of_columns(sample_path, data_shape):
    """test of sampled data fails if number of columns in sample does not correspond with configs"""
    sample_data(BASE_PATH)
    sample = pd.read_csv(sample_path, nrows=0)
    sample_columns_num = sample.shape[1]
    data_columns_num = data_shape[1]
    assert sample_columns_num == data_columns_num