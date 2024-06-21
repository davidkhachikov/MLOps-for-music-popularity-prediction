import math
import os
import pandas as pd
import hydra
import dvc.api

@hydra.main(config_path='../configs', config_name='main', version_base=None)
def sample_data(cfg=None):
    """
    Splits a dataset into equal batches and uploads each batch to a DVC remote repository.

    This function reads data from a local CSV file specified in the configuration,
    divides the data into batches according to the batch size defined in the configuration,
    and uploads each batch to a DVC remote repository.

    Parameters:
    - cfg (dict): Configuration dictionary containing details about the data source and batching.

    Returns:
    - None
    """
    # Load the data
    df = pd.read_csv(cfg.data_local.path)

    # Calculate the size of each batch
    batch_size = math.ceil(len(df) * cfg.data_local.sample_size)

    # Keep track of the number of versions uploaded
    num_versions_uploaded = 0

    for i in range(math.ceil(1 / cfg.data_local.sample_size)):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(df))

        # Extract the batch
        batch_df = df.iloc[start_index:end_index]

        # Define the path for saving the batch locally before uploading
        local_batch_path = f'./data/batch_{num_versions_uploaded}.csv'

        # Save the batch locally temporarily
        batch_df.to_csv(local_batch_path, index=False)

        # Upload the batch to the DVC remote repository
        dvc.api.add(local_batch_path,
                    path=f'data/{local_batch_path}',
                    repo=cfg.data_remote.repo,
                    rev=f'batch_{num_versions_uploaded}',
                    remote=cfg.data_remote.remote)

        # Remove the local batch file after upload
        os.remove(local_batch_path)

        if end_index == len(df):
            break

        # Increment the number of versions uploaded
        num_versions_uploaded += 1

if __name__ == "__main__":
    sample_data()
