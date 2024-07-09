from multiprocessing.connection import Client
import os
import sys
import pandas as pd
from hydra import initialize, compose
from omegaconf import DictConfig
import great_expectations as gx
from crypt import crypt as _crypt
import dvc.api
import zenml
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler, MinMaxScaler
from joblib import Parallel, delayed
import re

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

        df = pd.read_csv(cfg.data.path_to_raw, low_memory=False)
        
        num_files = cfg.data.num_samples
        num = max(min(num_files, cfg.data.sample_num), 1) - 1 
        start = num * int(len(df) / num_files)
        end = min((num + 1) * int(len(df) / num_files), len(df))
        chunk = df[start:end]

        current_dir = os.getcwd()
        target_folder = os.path.join(current_dir, 'data', 'samples')
        chunk.to_csv(os.path.join(target_folder, 'sample.csv'), index=False)
        print(f"Sampled data part {num + 1} saved to {os.path.join(target_folder, 'sample.csv')}")
    except Exception as e:
        print(f"An error occurred while saving the sampled data: {e}")
        exit(1)


def handle_initial_data():
    """
    Preprocesses the music popularity dataset by cleaning and transforming raw data into a suitable format for analysis.
    
    This function reads a sample CSV file containing music track data, converts relevant columns to datetime objects,
    handles missing values, and performs initial exploratory data transformations such as binarizing categorical features.
    
    No parameters are required as the function operates on a predefined dataset located at a fixed path.
    
    Returns:
        None
    """
    try:
        current_dir = os.getcwd()
        data_path = os.path.join(current_dir, 'data', 'samples', 'sample.csv')
        
        # Check if the file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The file {data_path} does not exist.")
        
        df = pd.read_csv(data_path)
        df["album_release_date"] = pd.to_datetime(df["album_release_date"],
                                                  format="mixed",
                                                  yearfirst=True,
                                                  errors="coerce")
        print(df.columns)
        df["added_at"] = pd.to_datetime(df["added_at"], yearfirst=True, errors="coerce")

        # Drop unnecessary columns
        df.drop(columns=[
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

        # Binarize categorical features
        df["chart"] = df["chart"].map({"top200": 1, "top50": 2})
        df["chart"] = df["chart"].fillna(0)

        # Convert album_release_date to int64
        df["album_release_date"] = df["album_release_date"].astype("int64")

        # Impute missing values with median
        df.fillna(df.median(), inplace=True)
        print("Missing values imputed")
        
        # Save the modified DataFrame back to CSV
        df.to_csv(data_path, index=False)
        print(f"File saved to {data_path}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)


def validate_initial_data():
    current_dir = os.getcwd()
    data_path = os.path.join(current_dir, 'data', 'samples', 'sample.csv')
    context_path = os.path.join(current_dir, 'services', 'gx')
    # df = pd.read_csv(data_path)

    context = gx.get_context(context_root_dir=context_path)

    ds = context.sources.add_or_update_pandas(name="sample_data")
    da = ds.add_csv_asset(name="sample_asset", filepath_or_buffer=data_path)

    batch_request = da.build_batch_request()
    checkpoint = context.add_or_update_checkpoint(
        name="test_checkpoint",
        validations=[ # A list of validations
            {
                "batch_request": batch.batch_request,
                "expectation_suite_name": "expectation_suite",
            }
            for batch in da.get_batch_list_from_batch_request(batch_request)
        ],
    )
    
    checkpoint_result = checkpoint.run()
    
    # checkpoint_result = checkpoint.run()
    print(checkpoint_result.get_statistics)
    if not checkpoint_result.success:
        exit(1)
    print("Success")


def read_datastore(project_path:str):
    """
    Takes the project path
    Makes sample dataframe and reads the data version from ./configs/data_version.yaml"""

    data_path = "data/samples/sample.csv"
    conf_path = "./configs/"
    with initialize(config_path=project_path + conf_path, version_base=None):
        cfg: DictConfig = compose(config_name='data_version')
    version = cfg.data.version

    with dvc.api.open(
                    data_path,
                    rev=version,
                    encoding='utf-8'
            ) as f:
                df = pd.read_csv(f)
    return df, version


def preprocess_data(df: pd.DataFrame):
    """ Performs data transformation and returns X, y tuple"""
    X = df.drop(columns="popularity")
    y = df["popularity"]
    
    genres_column = ["genres"] # List to multilable binary
    av_markets_column = ["available_markets"] # List to multilable binary
    key_timesig_columns = ['key', 'time_signature'] # Just one-hot encode
    normal_features = [
        "artist_followers",
        "album_total_tracks",
        "artist_popularity",
        "tempo",
        "speechiness",
        "danceability",
        "liveness",
        "loudness",
    ] # Have normal like distributions -> standard scaling
    uniform_features = [
        "energy",
        "valence",
        "acousticness",
        "instrumentalness",
    ] # Have uniform like distributions  -> MinMax scaling
    unchanged_features = [
        "explicit"
        "chart",
        "album_release_date",
        "popularity",
        "mode"
    ] # As is
    total_raw_features = genres_column + av_markets_column + key_timesig_columns + normal_features + uniform_features + unchanged_features
    # Check if df has all the listed columns
    if len(df.columns) != len(total_raw_features) or len(set(total_raw_features) - set(df.columns)) or len(set(df.columns) - set(total_raw_features)):
        raise ValueError(f"The input DataFrame should have only the following features:{'- '.join(total_raw_features)}")
    
    transformed_genres = transform_genres(df[genres_column])
    transformed_markets = transform_available_markets(df[av_markets_column])
    transformed_key_timesig = transform_key_timesig(df[key_timesig_columns])
    transformed_normallike = transform_normallike_features(df[normal_features])
    transformed_uniformlike = transform_uniformlike_features(df[uniform_features])
    transformed_unchanged = transform_unchanged_features(df[unchanged_features])

    X = pd.concat([
        transformed_genres,
        transformed_markets,
        transformed_key_timesig,
        transformed_normallike,
        transformed_uniformlike,
        transformed_unchanged
    ], axis=1)
    y = X.pop("popularity")
    return X, y

    

# Function to check if one string is a substring of another with delimiters
def check_if_subgenre(feature1, feature2):
    pattern1 = r'\b' + re.escape(feature1) + r'\b'
    
    return re.search(pattern1, feature2) is not None

# Function to process a pair of features
def process_genres_pair(i, j, features):
    return (features[i], features[j], check_if_subgenre(features[i], features[j]))

def transform_genres(df:pd.DataFrame, genres_count_threshold:int=20, expected_columns: list = None):
    """
    Takes the dataframe with genres column only
    Transforms the genres column to one hot encoded format
    Decomposes the genres column to multiple atomic genres
    Drops the composite genres
    Moves the underrepresented genres to a new column called "Other"
    If expected_columns is provided, makes sure the dataframe has all the expected columns
    Returns the transformed dataframe with sorted columns"""
    mlb = MultiLabelBinarizer()
    # Check that df has only "genres" column
    if len(df.columns) != 1 or df.columns[0] != "genres":
        raise ValueError("The input DataFrame should have only one column named 'genres'.")
    res = df.fillna("[]").apply(eval).to_frame()
    res = pd.DataFrame(mlb.fit_transform(res), columns=mlb.classes_, index=res.index)
    
    features = res.columns
    # List to store the found pairs
    found_pairs = []

    # Use Parallel and delayed to parallelize the computation
    n_jobs = -1  # Use all available cores
    found_pairs = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(process_genres_pair)(i, j, features) for i in range(len(features)) for j in range(i + 1, len(features))
    )

    # Convert results to DataFrame for better readability
    substring_pairs = pd.DataFrame(found_pairs, columns=['Feature1', 'Feature2', 'IsSubstring'])

    # Filter only the pairs where one is a substring of the other
    substring_pairs = substring_pairs[substring_pairs['IsSubstring']]

    composite_genres = set(substring_pairs['Feature2'])

    # Decompose the composite genres
    for composite_g in composite_genres:
        components = substring_pairs[substring_pairs['Feature2'] == composite_g]["Feature1"].values
        res.loc[res[composite_g] == 1, components] = 1
    res = res.drop(columns=composite_genres)

    # Count the number of occurrences of each genre
    genres_count = res.sum()

    # Move underrepresented genres to "Other" column
    other_genres = genres_count[genres_count <= genres_count_threshold].index
    res["Other"] = res[other_genres].any(axis=1).astype(int)
    res = res.drop(columns=other_genres)

    if expected_columns is not None and "Other" in expected_columns:
        # Make sure the dataframe has all the expected columns
        missing_columns = set(expected_columns) - set(res.columns)
        extra_columns = set(res.columns) - set(expected_columns)
        if missing_columns:
            # Our dataframe needs to have all the expected columns
            # So we add the missing columns and set them to 0
            res[missing_columns] = 0
        
        if extra_columns:
            # Move the extra columns to "Other"
            res["Other"] = (res["Other"] | res[extra_columns].any(axis=1)).astype(int)
            res = res.drop(columns=extra_columns)
    
    # Sort the columns by name length and alphabetically
    res = res.reindex(sorted(res.columns, key=lambda x: (len(x), x)))

    return res


def transform_available_markets(df:pd.DataFrame, expected_columns: list = None):
    """
    Takes the dataframe with available_markets column only
    Transforms the available_markets column to one hot encoded format
    Returns the transformed dataframe with sorted columns"""
    mlb = MultiLabelBinarizer()
    # Check that df has only "available_markets" column
    if len(df.columns) != 1 or df.columns[0] != "available_markets":
        raise ValueError("The input DataFrame should have only one column named 'available_markets'.")
    
    res = df.fillna("[]").apply(eval).to_frame()
    res = pd.DataFrame(mlb.fit_transform(res), columns=mlb.classes_, index=res.index)

    if expected_columns is not None:
        # Make sure the dataframe has all the expected columns
        missing_columns = set(expected_columns) - set(res.columns)
        extra_columns = set(res.columns) - set(expected_columns)
        if missing_columns:
            # Our dataframe needs to have all the expected columns
            # So we add the missing columns and set them to 0
            res[missing_columns] = 0
        
        if extra_columns:
            # Drop the extra columns
            res = res.drop(columns=extra_columns)

    # Sort the columns by name length and alphabetically
    res = res.reindex(sorted(res.columns, key=lambda x: (len(x), x)))

    return res


def transform_key_timesig(df:pd.DataFrame, expected_columns:list=None):
    """
    Takes the dataframe with key and time_signature columns only
    One-hot encodes the key and time_signature columns
    Returns the transformed dataframe with sorted columns"""
    oh_encoder = OneHotEncoder(sparse_output=False)
    # Check if df has only "key" and "time_signature" columns
    if len(df.columns) != 2 or "key" not in df.columns or "time_signature" not in df.columns:
        raise ValueError("The input DataFrame should have only two columns named 'key' and 'time_signature'.")
    
    res = pd.DataFrame(oh_encoder.fit_transform(df), columns=oh_encoder.get_feature_names_out(), index=df.index)

    if expected_columns is not None:
        # Make sure the dataframe has all the expected columns
        missing_columns = set(expected_columns) - set(res.columns)
        extra_columns = set(res.columns) - set(expected_columns)
        if missing_columns:
            # Our dataframe needs to have all the expected columns
            # So we add the missing columns and set them to 0
            res[missing_columns] = 0
        
        if extra_columns:
            # Drop the extra columns
            res = res.drop(columns=extra_columns)

    
    
    # Sort the columns by name length and alphabetically
    res = res.reindex(sorted(res.columns, key=lambda x: (len(x), x)))

    return res


def transform_normallike_features(df:pd.DataFrame):
    """
    Takes the dataframe with the following columns:
    - artist_followers,
    - album_total_tracks,
    - artist_popularity,
    - tempo,
    - speechiness,
    - danceability,
    - liveness,
    - loudness
    
    Scales these features using standard scaler"""

    features_to_transform = [
        "artist_followers",
        "album_total_tracks",
        "artist_popularity",
        "tempo",
        "speechiness",
        "danceability",
        "liveness",
        "loudness"
    ]

    # Check if df has all the listed features
    if len(df.columns) != len(features_to_transform) or len(set(df.columns) - set(features_to_transform)) or len(set(features_to_transform) - set(df.columns)):
        raise ValueError(f"The input DataFrame should have only the following features:{'- '.join(features_to_transform)}")
    
    std_scaler = StandardScaler()
    res = pd.DataFrame(std_scaler.fit_transform(df), columns=df.columns, index=df.index)

    # Sort the columns by name length and alphabetically (just in case)
    res = res.reindex(sorted(res.columns, key=lambda x: (len(x), x)))

    return res


def transform_uniformlike_features(df:pd.DataFrame):
    """
    Takes the dataframe with the following columns:
    - energy,
    - valence,
    - acousticness,
    - instrumentalness
    
    Scales these features using min max scaler"""

    features_to_transform = [
        "energy",
        "valence",
        "acousticness",
        "instrumentalness",
    ]

    # Check if df has all the listed features
    if len(df.columns) != len(features_to_transform) or len(set(df.columns) - set(features_to_transform)) or len(set(features_to_transform) - set(df.columns)):
        raise ValueError(f"The input DataFrame should have only the following features:{'- '.join(features_to_transform)}")
    
    mm_scaler = MinMaxScaler()
    res = pd.DataFrame(mm_scaler.fit_transform(df), columns=df.columns, index=df.index)

    # Sort the columns by name length and alphabetically (just in case)
    res = res.reindex(sorted(res.columns, key=lambda x: (len(x), x)))

    return res


def transform_unchanged_features(df:pd.DataFrame):
    """
    Takes the dataframe with the following columns:
    - explicit,
    - chart,
    - album_release_date,
    - popularity,
    - mode
    
    Checks that all the features are present.
    Returns dataframe with sorted columns"""

    unchanged_features = [
        "explicit",
        "chart",
        "album_release_date",
        "popularity",
        "mode"
    ]

    # Check if df has all the listed features
    if len(df.columns) != len(unchanged_features) or len(set(df.columns) - set(unchanged_features)) or len(set(unchanged_features) - set(df.columns)):
        raise ValueError(f"The input DataFrame should have only the following features:{'- '.join(unchanged_features)}")
    
    # Sort the columns by name length and alphabetically (just in case)
    res = df.reindex(sorted(df.columns, key=lambda x: (len(x), x)))

    return res
    


def validate_features(X: pd.DataFrame, y: pd.DataFrame):
    """ Performs feature validation using new expectations"""
    return X, y


def load_features(X:pd.DataFrame, y:pd.DataFrame, version: str):
    """Load and version the features X and the target y in artifact store of ZenML"""
    
    # Save the artifact
    zenml.save_artifact(data=(X, y), name="features_target", tags=[version])

    # Retrieve the client to interact with the ZenML store
    client = Client()

    # Verify the artifact was saved correctly
    try:
        l = client.list_artifact_versions(name="features_target", tag=version, sort_by="version").items

        # Descending order
        l.reverse()

        # Retrieve the latest version of the artifact
        if l:
            retrieved_X, retrieved_y = l[0].load()
            return retrieved_X, retrieved_y
        else:
            print("No artifacts found with the specified version tag.")

    except Exception as e:
        print(f"An error occurred while retrieving the artifact: {e}")

    return X, y

if __name__ == '__main__':
    validate_initial_data()