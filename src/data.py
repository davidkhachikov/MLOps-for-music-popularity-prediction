import os
import sys
import pandas as pd
from hydra import initialize, compose
from omegaconf import DictConfig
import great_expectations as gx
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
                df = pd.read_csv(f)

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

def validate_initial_data():
    current_dir = os.getcwd()
    data_path = os.path.join(current_dir, 'data', 'samples', 'sample.csv')
    df = pd.read_csv(data_path)

    context = gx.get_context(context_root_dir="../services")
    validator = context.sources.add_pandas("sample").read_dataframe(
        df
    )

    for column in df.columns:
        validator.expect_column_values_to_not_be_null(column)

    # artist_followers
    validator.expect_column_values_to_be_between("artist_followers", min_value=0)
    validator.expect_column_values_to_be_of_type("artist_followers", type_="NUMBER")

    # genres

    # album_total_tracks
    # artist_popularity
    validator.expect_column_values_to_be_between(
        "artist_popularity", min_value=0, max_value=100
    )
    validator.expect_column_values_to_be_of_type("artist_popularity", type_="float64")
    # explicit
    # tempo
    validator.expect_column_values_to_be_between("artist_followers", min_value=0)
    validator.expect_column_values_to_be_of_type("artist_followers", type_="float64")
    # chart
    validator.expect_column_values_to_be_in_set("chart", value_set=[0, 1, 2])
    # album_release_date

    # energy
    validator.expect_column_values_to_be_between(
        "energy", min_value=0, max_value=1
    )
    validator.expect_column_values_to_be_of_type("energy", type_="float64")
    # key
    validator.expect_column_values_to_be_in_set(
        "key", value_set=list(range(-1, 12))
    )
    # popularity
    validator.expect_column_values_to_be_between(
        "popularity", min_value=0, max_value=100
    )
    validator.expect_column_values_to_be_of_type("popularity", type_="float64")
    # available_markets
    # mode
    validator.expect_column_values_to_be_in_set(
        "mode", value_set=[0, 1]
    )
    # time_signature
    validator.expect_column_values_to_be_in_set(
        "time_signature", value_set=[0, 1, 2, 3, 4, 5, 6, 7]
    )
    # speechiness
    validator.expect_column_values_to_be_between(
        "speechiness", min_value=0, max_value=1
    )
    validator.expect_column_values_to_be_of_type("speechiness", type_="float64")
    # danceability
    validator.expect_column_values_to_be_between(
        "danceability", min_value=0, max_value=1
    )
    validator.expect_column_values_to_be_of_type("danceability", type_="float64")
    # valence
    validator.expect_column_values_to_be_between(
        "valence", min_value=0, max_value=1
    )
    validator.expect_column_values_to_be_of_type("valence", type_="float64")
    # acousticness
    validator.expect_column_values_to_be_between(
        "acousticness", min_value=0, max_value=1
    )
    validator.expect_column_values_to_be_of_type("acousticness", type_="float64")
    # liveness
    validator.expect_column_values_to_be_of_type("liveness", type_="float64")
    validator.expect_column_values_to_be_between("liveness", min_value=0)
    # instrumentalness
    validator.expect_column_values_to_be_between(
        "instrumentalness", min_value=0, max_value=1
    )
    validator.expect_column_values_to_be_of_type("instrumentalness", type_="float64")
    # loudness
    validator.expect_column_values_to_be_of_type("loudness", type_="float64")
    validator.expect_column_values_to_be_between("loudness", min_value=-60)

    validator.save_expectation_suite(discard_failed_expectations=False)
    checkpoint = context.add_or_update_checkpoint(
        name="my_checkpoint",
        validator=validator
    )
    checkpoint_result = checkpoint.run()
    if checkpoint_result.success:
        exit(0)
    exit(1)


def read_datastore(project_path:str):
    """
    Takes the project path
    Makes sample dataframe and reads the data version from ./configs/data_version.yaml"""

    data_path = "data/samples/sample.csv"
    df = pd.read_csv(project_path + data_path)
    version = "" #TODO
    
    return df, version


# def preprocess_data(df: pd.DataFrame):
#     """ Performs data transformation and returns X, y tuple"""
#     pass


def validate_features(X: pd.DataFrame, y: pd.DataFrame):
    """ Performs feature validation using new expectations"""
    pass


def load_features(X:pd.DataFrame, y:pd.DataFrame, version: str):
    """Load and version the features X and the target y in artifact store of ZenML"""
    pass



if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "preprocess_data":
        preprocess_data()
    elif len(sys.argv) == 3 and sys.argv[1] == "sample_data":
        sample_data(int(sys.argv[2]))
    elif len(sys.argv) == 2 and sys.argv[1] == 'validate_initial_data':
        validate_initial_data()
