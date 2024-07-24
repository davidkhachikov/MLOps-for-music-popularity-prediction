# src/validate.py

import os
import pickle

import numpy as np
from data import read_datastore
from model import load_features  # custom module
# from transform_data import transform_data  # custom module
from transform_data import transform_data
from utils import init_hydra  # custom module
from sklearn.metrics import mean_absolute_error, mean_squared_error
import argparse

import giskard
import mlflow
import pandas as pd
import pickle

def prepare_dataset(cfg, max_rows=1000):
    version = cfg.data.version
    df, version = read_datastore(version=version)
    df = df[:max_rows]
    target_column = cfg.data.target_features[0]
    dataset_name = f"{cfg.data.dataset_name}"
    
    giskard_dataset = giskard.Dataset(
        df=df,
        target=target_column,
        name=dataset_name
    )
    return giskard_dataset, df, version
    

def load_and_wrap_mlflow_model(model_path, feature_names, cfg):
    """
    Loads a model from using mlflow.
    """
    model = mlflow.pyfunc.load_model(model_path)
    def predict(raw_df):
        X = transform_data(
                            df = raw_df, 
                            cfg = cfg, 
                            return_df = False, 
                            only_transform = True,
                            only_X = True
                        )

        return model.predict(X)

    target_name = cfg.data.target_features[0]
    
    # Wrap the model with giskard.Model
    wrapped_model = giskard.Model(
        model=predict,
        feature_names=feature_names,
        target=target_name,
        model_type="regression"
    )
    
    return wrapped_model


def run_giskard_scan(giskard_model, giskard_dataset, model_name, model_alias, dataset_name, BASE_PATH):
    scan_results = giskard.scan(giskard_model, giskard_dataset)
    scan_results_path = f"{BASE_PATH}/reports/validation_results_{model_name}_{model_alias}_{dataset_name}.html"
    scan_results.to_html(scan_results_path)
    return scan_results


def create_and_run_test_suite(giskard_model, giskard_dataset, model_name, dataset_name, version, cfg):
    suite_name = f"regression_test_suite_{model_name}_{dataset_name}_{version}"
    test_suite = giskard.Suite(name=suite_name)

    def mae_test(model, dataset):
        threshold = cfg.mae_threshold
        y_true = dataset.df[dataset.target]
        y_pred = model.predict(dataset).raw
        mae = mean_absolute_error(y_true, y_pred)
        return giskard.TestResult(passed=mae <= threshold)

    # Correctly adding the test to the suite
    test_suite.add_test(mae_test, model=giskard_model, dataset=giskard_dataset, test_id="MAE_Test_ID")
    
    test_results = test_suite.run()
    return test_results


def validate_model(giskard_model, giskard_dataset, version, model_name, model_alias, BASE_PATH, cfg):
    test_results = create_and_run_test_suite(giskard_model, giskard_dataset, model_name, giskard_dataset.name, version, cfg)
    if test_results.passed:
        run_giskard_scan(giskard_model, giskard_dataset, model_name, model_alias, giskard_dataset.name, BASE_PATH)
    return test_results.passed


def main(model_path, model_alias, model_name):
    BASE_PATH = os.getenv('PROJECTPATH')
    cfg = init_hydra()
    giskard_dataset, df, version = prepare_dataset(cfg)

    feature_names = giskard_dataset.df.columns.tolist()  # Assuming this is how you access the DataFrame and its columns
    
    # Remove the target column from feature names if it's included
    target_name = giskard_dataset.target
    feature_names.remove(target_name)

    # Construct the full model path using the provided model_path and model_alias
    full_model_path = os.path.join(BASE_PATH, model_path)
    giskard_champion = load_and_wrap_mlflow_model(full_model_path, feature_names, cfg)
    if not validate_model(giskard_champion, giskard_dataset, version, model_name, model_alias, BASE_PATH, cfg):  # Pass model_name instead of 'model'
        raise Exception("Validation failed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate a model.')
    parser.add_argument('--model-path', type=str, default=f'{os.getenv("PROJECTPATH")}/models/champion', help='Path to the directory containing the model.')
    parser.add_argument('--model-alias', type=str, default='champion', help='Alias for the model')
    parser.add_argument('--model-name', type=str, default='hist_gradient_boost', help='Name of the model')  # Default value added here
    args = parser.parse_args()

    main(args.model_path, args.model_alias, args.model_name)
