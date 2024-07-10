from multiprocessing.connection import Client
import os
import sys
import numpy as np
import pandas as pd
from hydra import initialize, compose
from omegaconf import DictConfig
import great_expectations as gx
from crypt import crypt as _crypt
import dvc.api
import zenml
import ast
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, NotFittedError
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed
import re

from typing import Literal

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

    conf_path = "../configs/"
    with initialize(config_path=conf_path, version_base=None):
        cfg: DictConfig = compose(config_name='data_features.yaml')


    X = df.drop(columns="popularity")
    y = df["popularity"]

    
    # Define the base preprocessing pipeline for the multilabel columns
    multilabel_prep_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value="[]")), # Fill in missing values with empty lists
        ('to_list', FunctionTransformer(lambda x: pd.DataFrame(x).applymap(lambda k: ast.literal_eval(k)))) # Convert the string representation of lists to actual lists
    ])

    # Define the transformation pipeline for the genres column
    genres_transformer = Pipeline([
        ("preprocess", multilabel_prep_pipeline), # Preprocess the data
        ("decompose", GenreDecomposer()), # Replace composite genres with their atomic components
        ("encode", MultiHotEncoder()) # Binarize the genres
    ])

    # Define the transformation pipeline for other multilabel columns
    multilabel_transformer = Pipeline([
        ("preprocess", multilabel_prep_pipeline),
        ("encode", MultiHotEncoder())
    ])


    # Define the transformation pipeline for the categorical columns
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(sparse_output=False))
    ])

    # Define the transformation pipeline for the normal features
    normal_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Define the transformation pipeline for the uniform features
    uniform_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler())
    ])


    
    return X, y

    
class MultiHotEncoder(BaseEstimator, TransformerMixin):
    """Wraps `MultiLabelBinarizer` to allow for easy use in pipelines.
    """
    def __init__(self, input_format: Literal["pandas", "numpy"] = "pandas", handle_unknown="drop"):
        self.mlbs = {}
        self.input_format = input_format
        self._output_dtype = None
        self._output_format = "pandas"
        self._features_order = []
        self.classes_ = []
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        self.mlbs = {}
        self._features_order = []

        # if X has wrong format, raise an error
        if self.input_format == "pandas" and not isinstance(X, pd.DataFrame):
            raise ValueError("Input format is pandas, but X is not a pandas DataFrame")
        if self.input_format == "numpy" and not isinstance(X, np.ndarray):
            raise ValueError("Input format is numpy, but X is not a numpy array")
        
        # X must be 2D
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        elif len(X.shape) != 2:
            raise ValueError("Input must be 1D or 2D")
        
        if self.input_format == "pandas":
            for col in X.columns:
                self._features_order.append(col)
                self.mlbs[col] = MultiLabelBinarizer()
                self.mlbs[col].fit(X[col])
                self.classes_.extend(self.mlbs[col].classes_)
        else:
            for i in range(X.shape[1]):
                self._features_order.append(i)
                self.mlbs[i] = MultiLabelBinarizer()
                self.mlbs[i].fit(X[:, i])
                self.classes_.extend(self.mlbs[i].classes_)
        return self
    
    def transform(self, X):
        # if X has wrong format, raise an error
        if self.input_format == "pandas" and not isinstance(X, pd.DataFrame):
            raise ValueError("Input format is pandas, but X is not a pandas DataFrame")
        if self.input_format == "numpy" and not isinstance(X, np.ndarray):
            raise ValueError("Input format is numpy, but X is not a numpy array")
        
        # X must be 2D
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        elif len(X.shape) != 2:
            raise ValueError("Input must be 1D or 2D")
        
        # Check if the estimator has been fitted
        if not self.mlbs:
            raise NotFittedError("This MultiHotEncoder instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        
        if self.input_format == "pandas":
            # Check if the columns in X match the columns in the fitted data
            if not set(X.columns) == set(self.mlbs.keys()):
                raise ValueError("Columns in X do not match the columns in the fitted data")
            res = np.empty((X.shape[0], 0))
            for col in self._features_order:
                if self.handle_unknown == "drop":
                    # Some entries may contain unseen classes, so we need to filter them out
                    filtered = [set(entry).intersection(self.mlbs[col].classes_) for entry in X[col]]
                    res = np.concatenate([res, self.mlbs[col].transform(filtered)], axis=1)
                else:
                    res = np.concatenate([res, self.mlbs[col].transform(X[col])], axis=1)
        else:
            # Check if the number of columns in X match the number of columns in the fitted data
            if not len(self.mlbs) == X.shape[1]:
                raise ValueError("Number of columns in X does not match the number of columns in the fitted data")
            res = np.empty((X.shape[0], 0))
            for i in self._features_order:
                if self.handle_unknown == "drop":
                    # Some entries may contain unseen classes, so we need to filter them out
                    filtered = [set(entry).intersection(self.mlbs[i].classes_) for entry in X[:, i]]
                    res = np.concatenate([res, self.mlbs[i].transform(filtered)], axis=1, dtype=int)
                else:
                    res = np.concatenate([res, self.mlbs[i].transform(X[:, i])], axis=1, dtype=int)

        if self._output_format == "pandas":
            return pd.DataFrame(res, columns=self.classes_, dtype=int)
        else:
            return res
        
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    
    def set_output(self, *, transform=None):
        if transform is not None:
            self._output_format = transform
        return self

    def get_feature_names_out(self):
        if self.mlbs:
            return self.classes_
        else:
            return None
    
    
    
class GenreDecomposer(BaseEstimator, TransformerMixin):
    """Takes a column of lists of genres and replaces the composite genres with their atomic components."""
    def __init__(self, n_jobs=-1, input_format: Literal["pandas", "numpy"] = "pandas"):
        self.n_jobs = n_jobs
        self.composite_to_atomic = {}
        self.atomic_genres = []
        self._output_format = "pandas"
        self.input_format = input_format

    def fit(self, X, y=None):
        self.composite_to_atomic = {}
        self.atomic_genres = []

        # if X has wrong format, raise an error
        if self.input_format == "pandas" and not isinstance(X, pd.DataFrame):
            raise ValueError("Input format is pandas, but X is not a pandas DataFrame")
        if self.input_format == "numpy" and not isinstance(X, np.ndarray):
            raise ValueError("Input format is numpy, but X is not a numpy array")
        
        # X must be one column
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        elif len(X.shape) != 2 or X.shape[1] != 1:
            raise ValueError("Input must one column of lists of genres")
        
        features = set()

        # Get all the unique genres
        for entry in X.iloc[:, 0] if self.input_format == "pandas" else X[:, 0]:
            features.update(entry)

        features = sorted(features, key=len)

        # List to store the found pairs
        found_pairs = []

        # Use Parallel and delayed to parallelize the computation
        found_pairs = Parallel(n_jobs=self.n_jobs, backend="multiprocessing")(
            delayed(self.process_genres_pair)(i, j, features) for i in range(len(features)) for j in range(i + 1, len(features))
        )

        # Convert results to DataFrame for better readability
        substring_pairs = pd.DataFrame(found_pairs, columns=['Feature1', 'Feature2', 'IsSubstring'])

        # Filter only the pairs where one is a substring of the other
        substring_pairs = substring_pairs[substring_pairs['IsSubstring']]

        # Populate composite_to_atomic dictionary
        for _, row in substring_pairs.iterrows():
            if row['Feature2'] not in self.composite_to_atomic:
                self.composite_to_atomic[row['Feature2']] = []
            self.composite_to_atomic[row['Feature2']].append(row['Feature1'])
        
        self.atomic_genres = list(set(features) - set(self.composite_to_atomic.keys()))

        return self

    def transform(self, X):
        # if X has wrong format, raise an error
        if self.input_format == "pandas" and not isinstance(X, pd.DataFrame):
            raise ValueError("Input format is pandas, but X is not a pandas DataFrame")
        if self.input_format == "numpy" and not isinstance(X, np.ndarray):
            raise ValueError("Input format is numpy, but X is not a numpy array")
        
        # X must be one column
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        elif len(X.shape) != 2 or X.shape[1] != 1:
            raise ValueError("Input must one column of lists of genres")
        
        # Check if the estimator has been fitted
        if not self.composite_to_atomic:
            raise NotFittedError("This GenreDecomposer instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        # For each entry in the column, replace the composite genres with the atomic genres
        res = []
        for entry in X.iloc[:, 0] if self.input_format == "pandas" else X[:, 0]:
            new_entry = set()
            for genre in entry:
                if genre in self.atomic_genres:
                    new_entry.add(genre)
                else:
                    new_entry.update(self.composite_to_atomic.get(genre, []))
            res.append(list(new_entry))
        
        # Convert the result to the expected format
        # Result must be a 2D array
        if self._output_format == "pandas":
            return pd.DataFrame(np.array(res, dtype=list).reshape(-1, 1))
        else:
            return np.array(res, dtype=list).reshape(-1, 1)
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    
    def set_output(self, *, transform=None):
        if transform is not None:
            self._output_format = transform
        return self
    
    # Function to check if one string is a substring of another with delimiters
    def check_if_subgenre(self, feature1, feature2):
        pattern1 = r'\b' + re.escape(feature1) + r'\b'
        return re.search(pattern1, feature2) is not None

    # Function to process a pair of features
    def process_genres_pair(self, i, j, features):
        return (features[i], features[j], self.check_if_subgenre(features[i], features[j]))
    


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