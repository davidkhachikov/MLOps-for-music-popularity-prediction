import os
import subprocess
import pandas as pd
import numpy as np
from hydra import initialize, compose
from omegaconf import DictConfig
import dvc.api

def sample_data():
    """
    Loads a sample of music popularity data from a CSV file using DVC for version control and stores 
    it locally, split into multiple files as specified in the configuration.

    This function initializes Hydra to read configurations, then uses DVC to open and read a 
    specific version of the data file. It decides whether to fetch the data remotely or locally 
    based on the `remote` flag in the configuration. The sampled data is saved to local CSV files.

    Returns:
        None
    """
    try:
        with initialize(config_path="../configs", version_base=None):
            cfg: DictConfig = compose(config_name='main')

        if cfg.data.is_remote:
            # Remote sampling
            file = dvc.api.read(
                path=os.path.join(cfg.data.path),
                repo=cfg.data.repo,
                rev=cfg.data.version,
                remote=cfg.data.remote,
                encoding='utf-8'
            )
            df = pd.read_csv(file)
        else:
            # Local sampling
            with dvc.api.open(
                cfg.data.path,
                rev=cfg.data.version,
                encoding='utf-8'
            ) as f:
                df = pd.read_csv(f, low_memory=False)

        num_files = cfg.data.num_files
        chunks = np.array_split(df, num_files)

        current_dir = os.getcwd()
        target_folder = os.path.join(current_dir, 'data', 'samples')
        os.makedirs(target_folder, exist_ok=True)

        for i, chunk in enumerate(chunks, start=1):
            chunk.to_csv(os.path.join(target_folder, 'sample.csv'), index=False)
            full_command = ["pytest"] + ["-v"] + ["/tests"]
            result = subprocess.run(full_command, capture_output=True, text=True, check=True)           
            if result.returncode != 0:
                print("Tests failed.")
                print(result.stderr)
                continue
            else:
                print("Tests passed.")

            print(f"Sampled data part {i} saved to {os.path.join(target_folder, 'sample.csv')}")
    except Exception as e:
        print(f"An error occurred while saving the sampled data: {e}")

if __name__ == "__main__":
    sample_data()
