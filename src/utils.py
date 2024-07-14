from omegaconf import DictConfig
from hydra import initialize, compose

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