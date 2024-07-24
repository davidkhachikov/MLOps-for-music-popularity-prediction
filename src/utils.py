from omegaconf import DictConfig
from hydra import initialize, compose

def init_hydra():
    """
    Initializes the Hydra configuration system and loads the main configuration file.

    This function sets up the Hydra configuration system by specifying the path to the configuration files and loads the main configuration file. It returns the configuration as a DictConfig object, which can be used throughout the project to access configuration parameters.

    Returns:
        DictConfig: The main configuration object containing all the settings defined in the Hydra configuration files.

    Raises:
        FileNotFoundError: If the configuration file is not found at the specified path.
        Exception: Any other exceptions that occur during the initialization process.
    """
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