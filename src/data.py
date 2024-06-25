import os
import sys
import pandas as pd
from hydra import initialize, compose
from omegaconf import DictConfig
from sklearn.preprocessing import MultiLabelBinarizer
import dvc.api


def sample_data(num=0):
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
        start = num * int(len(df) / num_files)
        end = min((num + 1) * int(len(df) / num_files), len(df))
        chunk = df[start:end]

        current_dir = os.getcwd()
        target_folder = os.path.join(current_dir, 'data', 'samples')
        chunk.to_csv(os.path.join(target_folder, 'sample.csv'), index=False)
        print(f"Sampled data part {num + 1} saved to {os.path.join(target_folder, 'sample.csv')}")
    except Exception as e:
        print(f"An error occurred while saving the sampled data: {e}")


def preprocess_data():
    """
    Preprocesses the music popularity dataset by cleaning and transforming raw data into a suitable format for analysis.
    
    This function reads a sample CSV file containing music track data, converts relevant columns to datetime objects,
    handles missing values, and performs initial exploratory data transformations such as binarizing categorical features.
    
    No parameters are required as the function operates on a predefined dataset located at a fixed path.
    
    Returns:
        None
    """
    current_dir = os.getcwd()
    data_path = os.path.join(current_dir, 'data', 'samples', 'sample.csv')
    df = pd.read_csv(data_path)
    df["album_release_date"] = pd.to_datetime(df["album_release_date"],
                                              format="mixed",
                                              yearfirst=True,
                                              errors="coerce")
    df["added_at"] = pd.to_datetime(df["added_at"], yearfirst=True, errors="coerce")

    # Let's try to find some patterns in the data
    # First, let's restore the categorical data genres, available_markets
    pattern_df = df.copy()
    print("genres restored")
    pattern_df.drop(columns=["genres", "available_markets"], inplace=True)
    print(f"Dataset restored with {pattern_df.shape[1]} columns")

    # Now we will drop some columns
    pattern_df.drop(columns=[
        "track_id",  # unique identifier
        "streams",  # Too many missing values
        "track_artists",  # Too many missing values
        "added_at",  # Too many missing values and can be replaced by album_release_date
        "track_album_album",  # Too many missing values
        "duration_ms",  # Too many missing values
        "track_track_number",  # Too many missing values
        "rank",  # Too many missing values and dependent of chart
        "album_name",  # This is text data, and we do not do data transformation in this phase
        "region",  # Too many missing values
        "trend",  # Too many missing values and dependent of chart
        "name",  # This is text data, and we do not do data transformation in this phase
    ], inplace=True)
    print(f"Dataset reduced to {pattern_df.shape[1]} columns")

    # Replace nan chart with 0, top200 with 1 and top50 with 2
    pattern_df["chart"] = pattern_df["chart"].map({"top200": 1, "top50": 2})
    pattern_df["chart"] = pattern_df["chart"].fillna(0)
    print("Chart restored")

    # Convert album_release_date
    pattern_df["album_release_date"] = pattern_df["album_release_date"].astype("int64")

    # Now we will impute the missing values
    # Since the number of missing values is small, it is safe to use the median value
    pattern_df.fillna(pattern_df.median(), inplace=True)
    print("Missing values imputed")
    pattern_df.to_csv(data_path, index=False)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "preprocess_data":
        preprocess_data()
    elif len(sys.argv) == 3 and sys.argv[1] == "sample_data":
        sample_data(int(sys.argv[2]))
