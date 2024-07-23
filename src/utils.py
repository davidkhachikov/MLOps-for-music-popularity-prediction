from omegaconf import DictConfig
from hydra import initialize, compose
import dvc
import pandas as pd

def init_hydra():
    try:
        with initialize(config_path="../configs", version_base=None):
            cfg: DictConfig = compose(config_name='main')
        return cfg
    except FileNotFoundError as e:
        print(f"Configuration file not found: {e}")
        raise FileNotFoundError
    except Exception as e:
        # Catch any other exceptions that might occur
        print(f"An error occurred: {e}")
        raise

def get_test_raw(version=None):
    """
    Takes the project path
    Makes sample dataframe and reads the data version from ./configs/main.yaml
    """
    data_path = "data/raw/tracks.csv"
    cfg = init_hydra()
    if version is None:
        version = cfg.data.version[:-1]
        version += str(cfg.data.sample_num-1)
        
    with dvc.api.open(
                    data_path,
                    rev=version,
                    encoding='utf-8'
            ) as f:
                df = pd.read_csv(f, nrows=100)
    return df, version
