# test_validate_initial_data.py
from data import validate_initial_data, sample_data, handle_initial_data
import os

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