# test_handle_initial_data.py
import pytest
from src.data import handle_initial_data, sample_data
import pandas as pd
from src.utils import init_hydra, get_test_raw
import os

sample_path = "./data/samples/sample.csv"
data_path = "./data/raw/test_tracks.csv"
if not os.path.exists(data_path):
    data = get_test_raw()
    data.to_csv(data_path)
data_columns_num = pd.read_csv(data_path, nrows=0, low_memory=False).shape[1]

BASE_PATH = os.getenv('PROJECTPATH')

@pytest.mark.parametrize("sample_path", [sample_path])
def test_handle_initial_data_columns(sample_path):
    cfg = init_hydra()
    sample_data(BASE_PATH, data_path)
    handle_initial_data(BASE_PATH)
    sample_columns_num = pd.read_csv(sample_path, nrows=0).shape[1]
    assert sample_columns_num == data_columns_num - len(cfg.data.low_features_number)

@pytest.mark.parametrize("sample_path", [sample_path])
def test_handle_initial_data_missing_values(sample_path):
    sample_data(BASE_PATH, data_path)
    handle_initial_data(BASE_PATH)
    sample = pd.read_csv(sample_path)
    assert not sample.isnull().values.any(), "DataFrame contains NaN values."
    assert not sample.isna().values.all(), "All values are None."