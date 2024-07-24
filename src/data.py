from zenml.client import Client
import os
import numpy as np
import pandas as pd
import great_expectations as gx
import dvc.api
import zenml
import ast
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import NotFittedError
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import remove_stopwords
from utils import init_hydra
import joblib
import re

from typing import Literal

BASE_PATH = os.getenv('PROJECTPATH')

#######################
# Data Acquisition #

def sample_data(project_path=BASE_PATH):
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

        df = pd.read_csv(cfg.data.path_to_raw, low_memory=False)
        
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
        version = cfg.data.version
        
    with dvc.api.open(
                    data_path,
                    rev=version,
                    encoding='utf-8'
            ) as f:
                df = pd.read_csv(f, low_memory=False)
    return df, version

#######################
# Data Preprocessing #
        
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
    clean_names_concat = clean_names_concat.apply(remove_stopwords)
    names2vec_model = Word2Vec(clean_names_concat.str.split(), vector_size=10, window=5, min_count=1, workers=4)
    names2vec_model_path = os.path.join(transformers_dir, 'names2vec_model.sav')
    with open(names2vec_model_path, 'wb') as f:
        joblib.dump(names2vec_model, f)  


def handle_uniform_features(features: pd.DataFrame, transformers_dir):
    """Handles the uniform features in the dataset."""
    
    try:
        # Load the sklearn pipeline from 
        uniform_pipeline = joblib.load(os.path.join(transformers_dir, f"uniform_pipeline.sav"))
    except FileNotFoundError:
        # Define the transformers for the uniform features
        uniform_transformers = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler())
        ]

        # Create the pipeline
        uniform_pipeline = Pipeline(uniform_transformers)
        uniform_pipeline.fit(features)
        joblib.dump(uniform_pipeline, os.path.join(transformers_dir, f"uniform_pipeline.sav"))

    return pd.DataFrame(uniform_pipeline.transform(features), columns=features.columns)


def handle_normal_features(features: pd.DataFrame, transformers_dir):
    """Handles the normal features in the dataset."""
    
    try:
        # Load the sklearn pipeline from 
        normal_pipeline = joblib.load(os.path.join(transformers_dir, f"normal_pipeline.sav"))
    except FileNotFoundError:
        # Define the transformers for the normal features
        normal_transformers = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]

        # Create the pipeline
        normal_pipeline = Pipeline(normal_transformers)
        normal_pipeline.fit(features)
        joblib.dump(normal_pipeline, os.path.join(transformers_dir, f"normal_pipeline.sav"))

    return pd.DataFrame(normal_pipeline.transform(features), columns=features.columns)


def handle_onehot_features(features: pd.DataFrame, transformers_dir):
    """Handles the one-hot encoded features in the dataset."""
    
    try:
        # Load the sklearn pipeline from 
        onehot_pipeline = joblib.load(os.path.join(transformers_dir, f"onehot_pipeline.sav"))
    except FileNotFoundError:
        # Define the transformers for the one-hot encoded features
        onehot_transformers = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]

        # Create the pipeline
        onehot_pipeline = Pipeline(onehot_transformers)
        onehot_pipeline.fit(features)
        joblib.dump(onehot_pipeline, os.path.join(transformers_dir, f"onehot_pipeline.sav"))

    return pd.DataFrame(onehot_pipeline.transform(features), columns=onehot_pipeline.named_steps["onehot"].get_feature_names_out(features.columns))            


# Define the cyclical feature transformers
def sin_transformer(data, period):
    return np.sin(data.astype(float) / period * 2 * np.pi)

def cos_transformer(data, period):
    return np.cos(data.astype(float) / period * 2 * np.pi)


def handle_date_features(features: pd.DataFrame, transformers_dir):
    """Handles the date features in the dataset."""

    # Split each date feature into year, month, day, and day of the week
    res = pd.DataFrame(columns = [f"{col}_{i}" for col in features.columns for i in ["year", "month_sin", "month_cos", "day_sin", "day_cos", "weekday_sin", "weekday_cos"]], index=features.index)
    for col in features.columns:
        # Convert the column to datetime if it's not already
        date_col = pd.to_datetime(features[col], yearfirst=True, errors="coerce")
        # Extract year, month, day, and weekday
        res[f"{col}_year"] = date_col.dt.year
        res[f"{col}_month_sin"] = sin_transformer(date_col.dt.month, 12)
        res[f"{col}_month_cos"] = cos_transformer(date_col.dt.month, 12)
        res[f"{col}_day_sin"] = sin_transformer(date_col.dt.day, 31)
        res[f"{col}_day_cos"] = cos_transformer(date_col.dt.day, 31)
        res[f"{col}_weekday_sin"] = sin_transformer(date_col.dt.weekday, 7)
        res[f"{col}_weekday_cos"] = cos_transformer(date_col.dt.weekday, 7)
    return res

def handle_names_features(features: pd.DataFrame, transformers_dir):
    """Handles the names features in the dataset."""
    
    # Concatenate the names features
    # Clean the text
    clean_names_concat = features.apply(lambda x: " ".join(x), axis=1)
    clean_names_concat = clean_names_concat.apply(lambda x: re.sub(r'\W+', ' ', x).lower())

    # Load the Word2Vec model
    try:
        names2vec_model = joblib.load(os.path.join(transformers_dir, 'names2vec_model.sav'))
    except FileNotFoundError:
        names2vec_model = Word2Vec(clean_names_concat.str.split(), vector_size=10, window=5, min_count=1, workers=4)
        with open(os.path.join(transformers_dir, 'names2vec_model.sav'), 'wb') as f:
            joblib.dump(names2vec_model, f)

    # Transform the names features
    names_features = averaged_word_vectorizer(clean_names_concat.str.split(), names2vec_model, names2vec_model.vector_size)
    return pd.DataFrame(names_features, columns=[f"n_vec_{i}" for i in range(names2vec_model.vector_size)])


def handle_genres_features(features: pd.DataFrame, transformers_dir):
    """Handles the genres features in the dataset."""
    
    # Clean the genres
    clean_genres = features.apply(lambda x: re.sub(r'\W+', ' ', x).lower())
    
    # Load the Word2Vec model
    try:
        genres2vec_model = joblib.load(os.path.join(transformers_dir, 'genres2vec_model.sav'))
    except FileNotFoundError:
        genres2vec_model = Word2Vec(clean_genres.str.split(), vector_size=10, window=5, min_count=1, workers=4)
        with open(os.path.join(transformers_dir, 'genres2vec_model.sav'), 'wb') as f:
            joblib.dump(genres2vec_model, f)

    # Transform the genres features
    genres_features = averaged_word_vectorizer(clean_genres.str.split(), genres2vec_model, genres2vec_model.vector_size)
    return pd.DataFrame(genres_features, columns=[f"g_vec_{i}" for i in range(genres2vec_model.vector_size)])


def handle_multilabel_features(features: pd.DataFrame, transformers_dir):
    """Handles the multilabel features in the dataset."""
    
    res = pd.DataFrame(index=features.index)

    for col in features.columns:
        # Fill missing values with empty lists
        features[col] = features[col].apply(lambda x: re.sub(r'\W+', ' ', x)).apply(str.split)

        # Load the MultiLabelBinarizer
        try:
            mlb = joblib.load(os.path.join(transformers_dir, f"{col}_mlb.sav"))
        except FileNotFoundError:
            mlb = MultiLabelBinarizer()
            mlb.fit(features[col])
            joblib.dump(mlb, os.path.join(transformers_dir, f"{col}_mlb.sav"))
        
        # Transform the multilabel features
        transformed = mlb.transform(features[col])
        res = pd.concat([res, pd.DataFrame(transformed, columns=[f"{col}_{i}" for i in mlb.classes_])], axis=1)

    return res


def convert_types(features: pd.DataFrame, expected_type):
    """Converts the features to the expected type"""
    
    # Convert the features to the expected type
    for col in features.columns:
        features[col] = features[col].astype(expected_type)
    return features



def preprocess_data(df: pd.DataFrame, only_X=False):
    """ Performs data transformation and returns X, y tuple"""

    cfg = init_hydra()

    # Splitting the data into features and targets
    
    X = df.drop(columns=cfg.data.target_features, errors='ignore')
    if not only_X:
        y = df[cfg.data.target_features]

    # Create the transformers directory if it does not exist
    transformers_dir = os.path.join(BASE_PATH, "data", "transformers")
    if not os.path.exists(transformers_dir):
        os.makedirs(transformers_dir)

    # Handle the uniform features
    X_uniform = handle_uniform_features(X[cfg.data.uniform_features], transformers_dir)

    # Handle the normal features
    X_normal = handle_normal_features(X[cfg.data.normal_features], transformers_dir)

    # Handle the one-hot encoded features
    X_onehot = handle_onehot_features(X[cfg.data.categorical_features], transformers_dir)

    # Handle the date features
    X_date = handle_date_features(X[cfg.data.timedate_features], transformers_dir)

    # Handle the names features
    X_names = handle_names_features(X[cfg.data.text_features], transformers_dir)

    # Handle the genres features
    X_genres = handle_genres_features(X[cfg.data.genres_feature], transformers_dir)

    # Convert to bool
    X_bool = convert_types(X[cfg.data.convert_to_bool], bool)

    # Handle the multilabel features
    X_multilabel = handle_multilabel_features(X[cfg.data.multilabel_features], transformers_dir)
    if only_X:
        return pd.concat([
        X_uniform, 
        X_normal, 
        X_onehot, 
        X_date, 
        X_names, 
        X_genres, 
        X_bool, 
        X_multilabel
        ], axis=1)

    return pd.concat([
        X_uniform, 
        X_normal, 
        X_onehot, 
        X_date, 
        X_names, 
        X_genres, 
        X_bool, 
        X_multilabel
        ], axis=1), y


    


#######################
# Feature Validation #

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

#######################
# Feature Loading #

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
