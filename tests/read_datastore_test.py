# test_read_datastore.py
import pytest
from src.data import read_datastore
from unittest.mock import MagicMock
from io import StringIO
import pandas as pd
from src.utils import init_hydra

@pytest.fixture
def mock_dvc_open(mocker):
    """Fixture to mock dvc.api.open and init_hydra to reflect main.yaml configuration."""
    # Mock dvc.api.open to return a sample CSV content
    mocker.patch('dvc.api.open', side_effect=lambda *args, **kwargs: StringIO("mock,data\n1,2"))
    # Mock init_hydra to return a configuration matching main.yaml
    mocker.patch('utils.init_hydra', return_value=MagicMock(data={
        'num_samples': 5,
        'path_to_raw': './data/raw/test_tracks.csv',
        'remote': 'mlops_remote',
        'repo': '.',
        'sample_num': 0,
        'version': 'AIRFLOW2.0'
    }))

@pytest.mark.usefixtures("mock_dvc_open")
def test_read_datastore_success():
    """Test read_datastore succeeds with correct main.yaml and data."""
    df, version = read_datastore()
    cfg = init_hydra()
    
    assert isinstance(df, pd.DataFrame)
    assert version == 'AIRFLOW2.'+str(cfg.data.sample_num-1)