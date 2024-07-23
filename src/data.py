from zenml.client import Client
import os
import numpy as np
import pandas as pd
import great_expectations as gx
import dvc.api
import zenml
import ast
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import NotFittedError
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from gensim.models import Word2Vec
from src.utils import init_hydra
import joblib
import re

from typing import Literal

BASE_PATH = os.getenv('PROJECTPATH')

def sample_data(project_path=BASE_PATH, path_to_raw=None):
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
        cfg = init_hydra()
        if path_to_raw is None:
            df = pd.read_csv(cfg.data.path_to_raw, low_memory=False)
        else:
            df = pd.read_csv(path_to_raw, low_memory=False)
        
        num_files = cfg.data.num_samples
        num = cfg.data.sample_num % num_files
        start = num * int(len(df) / num_files)
        end = min((num + 1) * int(len(df) / num_files), len(df))
        chunk = df[start:end]

        target_folder = os.path.join(project_path, 'data', 'samples')
        chunk.to_csv(os.path.join(target_folder, 'sample.csv'), index=False)
        print(f"Sampled data part {num + 1} saved to {os.path.join(target_folder, 'sample.csv')}")
    except Exception as e:
        print(f"An error during saving sample: {e}")
        raise


def handle_initial_data(project_path=BASE_PATH):
    """
    Preprocesses the music popularity dataset by cleaning and transforming raw data into a suitable format for analysis.
    
    This function reads a sample CSV file containing music track data, converts relevant columns to datetime objects,
    handles missing values, and performs initial exploratory data transformations such as binarizing categorical features.
    
    No parameters are required as the function operates on a predefined dataset located at a fixed path.
    
    Returns:
        None
    """
    try:
        data_path = os.path.join(project_path, 'data', 'samples', 'sample.csv')
        
        # Check if the file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The file {data_path} does not exist.")
        
        cfg = init_hydra()

        df = pd.read_csv(data_path)
        # Drop unnecessary columns
        df.drop(columns=cfg.data.low_features_number, inplace=True)

        # preprocess datetime features
        for feature in cfg.data.timedate_features:
            df[feature] = pd.to_datetime(df[feature], yearfirst=True, errors="coerce")
            df[feature].fillna(pd.Timestamp('1970-01-01'), inplace=True)

        for feature in cfg.data.missing_list:
            df[feature] = df[feature].apply(lambda d: d if d is not np.nan else [])

        for feature in cfg.data.missing_strings:
            df[feature] = df[feature].fillna(" ")
        
        # Binarize categorical features
        df["chart"] = df["chart"].map({"top200": 1, "top50": 2})
        df["chart"] = df["chart"].fillna(0)

        # Impute missing values with median
        df.fillna(df.median(), inplace=True)
        print("Missing values imputed")
        
        # Save the modified DataFrame back to CSV
        df.to_csv(data_path, index=False)
        print(f"File saved to {data_path}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

def validate_initial_data(project_path=BASE_PATH):
    try:
        data_path = os.path.join(project_path, 'data', 'samples', 'sample.csv')

        # Check if the sample exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The sample {data_path} does not exist.")

        context_path = os.path.join(project_path, 'services', 'gx')

        # Check if the context exists
        if not os.path.exists(context_path):
            raise FileNotFoundError(f"The context {context_path} does not exist.")
        
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

        # Outputs failed general statistics about validation
        print(checkpoint_result.get_statistics)
        if not checkpoint_result.success:
            raise Exception()
        print("Success")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise



def read_datastore(version=None):
    """
    Takes the project path
    Makes sample dataframe and reads the data version from ./configs/main.yaml
    """
    data_path = "data/samples/sample.csv"
    cfg = init_hydra()
    if version is None:
        version = cfg.data.version[:-1]
        version += str(cfg.data.sample_num-1)
        
    with dvc.api.open(
                    data_path,
                    rev=version,
                    encoding='utf-8'
            ) as f:
                df = pd.read_csv(f, low_memory=False)
    return df, version


def preprocess_data(df: pd.DataFrame):
    """ Performs data transformation and returns X, y tuple"""

    cfg = init_hydra()

    # Splitting the data into features and targets
    X = df.drop(columns=cfg.data.target_features)
    y = df[cfg.data.target_features]

    # Fit transformers if they don't exist
    transformers_dir = os.path.join(BASE_PATH, 'models', 'transformers')
    genres2vec_model_path = os.path.join(transformers_dir, 'genres2vec_model.pkl')

    if not os.path.exists(genres2vec_model_path):
        fit_transformers(X, cfg, transformers_dir)

    # Transform the data
    X_transformed = transform_data(X, cfg, transformers_dir)

    return X_transformed, y


def apply_literal_eval(df: pd.DataFrame):
    """ Applies literal_eval to all values in the DataFrame"""
    for col in df.columns:
        df[col] = df[col].apply(ast.literal_eval)
    return df

def decompose_dates(df: pd.DataFrame):
    """ Decomposes the date features into year, month, day, and weekday"""
    res = pd.DataFrame(columns=["year", "month", "day", "weekday"])
    df = df.squeeze()
    df = pd.to_datetime(df)
    res["year"] = df.dt.year
    res["month"] = df.dt.month
    res["day"] = df.dt.day
    res["weekday"] = df.dt.weekday

    return res

def get_column_transformer(cfg):
    # Define the base preprocessing pipeline for the multilabel columns
    multilabel_prep_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value="[]")), # Fill in missing values with empty lists if any
        ('to_list', FunctionTransformer(apply_literal_eval)) # Convert the string representation of lists to actual lists
    ])

    # Define the transformation pipeline for other multilabel columns
    multilabel_transformer = Pipeline([
        ("preprocess", multilabel_prep_pipeline),
        ("encode", MultiHotEncoder())
    ])


    # Define the transformation pipeline for the categorical columns
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(sparse_output=False, dtype=bool, handle_unknown='ignore'))
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

    # Define the transformation pipeline for the date features
    date_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=pd.Timestamp('1970-01-01'))),
        # all dates are Year-Month-Day. Split them into 4 columns
        ("year_month_day_weekday", FunctionTransformer(decompose_dates)),
        # Convert to cyclical features
        ('cyclic_convert', ColumnTransformer(
            transformers=[
                ('month_sin', sin_transformer(12), ['month']),
                ('month_cos', cos_transformer(12), ['month']),
                ('day_sin', sin_transformer(31), ['day']),
                ('day_cos', cos_transformer(31), ['day']),
                ('weekday_sin', sin_transformer(7), ['weekday']),
                ('weekday_cos', cos_transformer(7), ['weekday']),
                ('year', 'passthrough', ['year'])
            ], 
            remainder='drop'
        )),
        ('rename', FunctionTransformer(lambda x: x.rename(columns={
            "month_sin__month": "month_sin",
            "month_cos__month": "month_cos",
            "day_sin__day": "day_sin",
            "day_cos__day": "day_cos",
            "weekday_sin__weekday": "weekday_sin",
            "weekday_cos__weekday": "weekday_cos",
            "year__year": "year"
        }))),
    ])

    # Defien the column transformer
    column_transformer = ColumnTransformer(
        transformers=[
            ('multilabel', multilabel_transformer, list(cfg.data.multilabel_features)),
            ('categorical', categorical_transformer, list(cfg.data.categorical_features)),
            ('normal', normal_transformer, list(cfg.data.normal_features)),
            ('uniform', uniform_transformer, list(cfg.data.uniform_features)),
            ('dates', date_transformer, list(cfg.data.timedate_features)),
            ('int', FunctionTransformer(lambda x: x.astype(int)), list(cfg.data.ordinal_features)),
            ('bool', FunctionTransformer(lambda x: x.astype(bool)), list(cfg.data.convert_to_bool))
        ],
        remainder='drop',
        verbose=True
    )

    column_transformer.set_output(transform="pandas")

    return column_transformer



def fit_transformers(features: pd.DataFrame, cfg, transformers_dir):
    """Fits the transformers on the initial data sample and saves them as artifacts."""        
    # Fit the Word2Vec model
    clean_genres = features[cfg.data.genres_feature].apply(lambda x: re.sub(r'\W+', ' ', x).lower())
    genres2vec_model = Word2Vec(clean_genres.str.split(), vector_size=10, window=5, min_count=1, workers=4)
    genres2vec_model_path = os.path.join(transformers_dir, 'genres2vec_model.sav')
    with open(genres2vec_model_path, 'wb') as f:
        joblib.dump(genres2vec_model, f)

    clean_names_concat = features[cfg.data.text_features].apply(lambda x: " ".join(x), axis=1)
    clean_names_concat = clean_names_concat.apply(lambda x: re.sub(r'\W+', ' ', x).lower())
    names2vec_model = Word2Vec(clean_names_concat.str.split(), vector_size=10, window=5, min_count=1, workers=4)
    names2vec_model_path = os.path.join(transformers_dir, 'names2vec_model.sav')
    with open(names2vec_model_path, 'wb') as f:
        joblib.dump(names2vec_model, f)  


def transform_data(features: pd.DataFrame, cfg, transformers_dir):
    """Loads the fitted transformers and applies them to new data samples."""

    # Load the fitted column transformer
    column_transformer = get_column_transformer(cfg)
    
    # Load the fitted Word2Vec models
    genres2vec_model_path = os.path.join(transformers_dir, 'genres2vec_model.sav')
    with open(genres2vec_model_path, 'rb') as f:
        genres2vec_model = joblib.load(f)

    names2vec_model_path = os.path.join(transformers_dir, 'names2vec_model.sav')
    with open(names2vec_model_path, 'rb') as f:
        names2vec_model = joblib.load(f)

    # Apply the column transformer
    X_transformed = column_transformer.fit_transform(features)

    # Apply the Word2Vec models
    clean_genres = features[cfg.data.genres_feature].apply(lambda x: re.sub(r'\W+', ' ', x).lower())
    genres2vec_features = averaged_word_vectorizer(clean_genres.str.split(), genres2vec_model, 10)
    genres2vec_features = pd.DataFrame(genres2vec_features, columns=[f"g_vec_{i}" for i in range(10)])

    clean_names_concat = features[cfg.data.text_features].apply(lambda x: " ".join(x), axis=1)
    clean_names_concat = clean_names_concat.apply(lambda x: re.sub(r'\W+', ' ', x).lower())
    names2vec_features = averaged_word_vectorizer(clean_names_concat.str.split(), names2vec_model, 10)
    names2vec_features = pd.DataFrame(names2vec_features, columns=[f"n_vec_{i}" for i in range(10)])
    
    # Concatenate the transformed features with the Word2Vec features
    X_final = pd.concat([X_transformed, genres2vec_features, names2vec_features], axis=1)
    
    return X_final


# Define the cyclical feature transformers
def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x.astype(float) / period * 2 * np.pi))

def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x.astype(float) / period * 2 * np.pi))

    
class MultiHotEncoder(BaseEstimator, TransformerMixin):
    """Wraps `MultiLabelBinarizer` to allow for easy use in pipelines.
    """
    def __init__(self, input_format: Literal["pandas", "numpy"] = "pandas", handle_unknown="drop"):
        self.mlbs = {}
        self.input_format = input_format
        self._output_dtype = bool
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
            res = np.empty((X.shape[0], 0), dtype=self._output_dtype)
            for i in self._features_order:
                if self.handle_unknown == "drop":
                    # Some entries may contain unseen classes, so we need to filter them out
                    filtered = [set(entry).intersection(self.mlbs[i].classes_) for entry in X[:, i]]
                    res = np.concatenate([res, self.mlbs[i].transform(filtered)], axis=1, dtype=self._output_dtype)
                else:
                    res = np.concatenate([res, self.mlbs[i].transform(X[:, i])], axis=1, dtype=self._output_dtype)

        if self._output_format == "pandas":
            return pd.DataFrame(res, columns=self.classes_, dtype=self._output_dtype)
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
        
# Create a function to average the word vectors
def average_word_vectors(words, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features,), dtype="float64")
    nwords = 0.
    
    for word in words:
        if word in vocabulary:
            nwords = nwords + 1
            feature_vector = np.add(feature_vector, model.wv[word])
    
    if nwords:
        feature_vector = np.divide(feature_vector, nwords)
    
    return feature_vector

# Create a function to iterate over the dataset
def averaged_word_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index_to_key)
    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features) for tokenized_sentence in corpus]
    return np.array(features)



def validate_features(X: pd.DataFrame, y: pd.DataFrame):
    """ Performs feature validation using new expectations"""
    
    context = gx.get_context(context_root_dir=f"{BASE_PATH}/services/gx")
    data_asset = context.get_datasource("features").get_asset("features_dataframe")
    batch_request = data_asset.build_batch_request(dataframe=X)

    checkpoint = context.add_or_update_checkpoint(
        name="features_validation",
        validations=[
            {
                "batch_request": batch.batch_request,
                "expectation_suite_name": "features_expectations"
            } for batch in data_asset.get_batch_list_from_batch_request(batch_request)
        ]
    )

    checkpoint_result = checkpoint.run()
    
    if not checkpoint_result.success:
        print("Validation failed")
        print(checkpoint_result.get_statistics())
        exit(1)
    return X, y


def load_features(X: pd.DataFrame, y: pd.DataFrame, version: str):
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
