# test_handle_initial_data.py
import pytest
from data import handle_initial_data, sample_data
import pandas as pd
from utils import init_hydra
import os

sample_path = "./data/samples/sample.csv"
data_path = "./data/raw/tracks.csv"

data_columns_num = pd.read_csv(data_path, nrows=0, low_memory=False).shape[1]

BASE_PATH = os.getenv('PROJECTPATH')

@pytest.mark.parametrize("sample_path", [sample_path])
def test_handle_initial_data_columns(sample_path):
    cfg = init_hydra()
    sample_data(BASE_PATH)
    handle_initial_data(BASE_PATH)
    sample_columns_num = pd.read_csv(sample_path, nrows=0).shape[1]
    assert sample_columns_num == data_columns_num - len(cfg.data.low_features_number)

@pytest.mark.parametrize("sample_path", [sample_path])
def test_handle_initial_data_missing_values(sample_path):
    sample_data(BASE_PATH)
    handle_initial_data(BASE_PATH)
    sample = pd.read_csv(sample_path)
    assert not sample.isnull().values.any(), "DataFrame contains NaN values."
    assert not sample.isna().values.all(), "All values are None."