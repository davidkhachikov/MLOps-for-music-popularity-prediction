# test_validate_initial_data.py
from src.data import validate_initial_data, sample_data, handle_initial_data
import os
from src.utils import get_test_raw

data_path = "./data/raw/test_tracks.csv"
if not os.path.exists(data_path):
    data = get_test_raw()
    data.to_csv(data_path)

BASE_PATH = os.getenv('PROJECTPATH')
sample_data(BASE_PATH)

def test_validate_initial_data_on_malformed():
    sample_data(BASE_PATH)
    try:
        validate_initial_data(BASE_PATH)
        assert False
    except Exception as _:
        assert True

def test_validate_initial_data_on_good():
    sample_data(BASE_PATH)
    handle_initial_data(BASE_PATH)
    try:
        validate_initial_data(BASE_PATH)
        assert True
    except Exception as _:
        assert False