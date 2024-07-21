# src/validate.py

import os
import pickle

import numpy as np
from model import load_features  # custom module
# from transform_data import transform_data  # custom module
from utils import init_hydra  # custom module
from sklearn.metrics import mean_squared_error

import giskard
import mlflow
import pandas as pd

def prepare_dataset(cfg):
    version = cfg.data.version
    X, y = load_features(name="features_target", version=version)
    target_column = cfg.data.target_features[0]
    dataset_name = f"{cfg.data.dataset_name}.{version}"
    
    # Assuming X and y are numpy arrays or similar structures, concatenate them into a DataFrame
    df = pd.DataFrame(X)
    df[target_column] = y
    
    giskard_dataset = giskard.Dataset(
        df=df,
        target=target_column,
        name=dataset_name,
    )
    return giskard_dataset, df, version


def load_and_wrap_pickle_model(file_path, feature_names, cfg):
    """
    Loads a model from a .pkl file.
    """
    with open(file_path, 'rb') as file:
        model = pickle.load(file)

    target_name = cfg.data.target_features[0]
    
    # Wrap the model with giskard.Model
    wrapped_model = giskard.Model(
        model=model,
        feature_names=feature_names,
        target=target_name,
        model_type="regression"
    )
    
    return wrapped_model
    

def load_and_wrap_mlflow_model(model_uri, feature_names, cfg):
    """
    Loads a scikit-learn model from an MLflow model URI and wraps it with Giskard's Model class.
    
    :param model_uri: URI of the scikit-learn model to load.
    :param feature_names: List of feature names.
    :param cfg: Configuration object or dictionary containing model configuration.
    :return: A Giskard model wrapper around the loaded scikit-learn model.
    """
    # Load the scikit-learn model using MLflow
    model = mlflow.sklearn.load_model(model_uri)
    
    target_name = cfg.data.target_features[0]
    
    # Define a prediction function using the loaded model's predict method
    def predict_function(X):
        # Convert input features to the expected format (if necessary)
        # This step depends on how your model expects inputs
        return model.predict(X)
    
    wrapped_model = giskard.Model(
        model=predict_function,  # Use the prediction function instead of the model object
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


def create_and_run_test_suite(giskard_model, giskard_dataset, model_name, dataset_name, version, threshold):
    suite_name = f"regression_test_suite_{model_name}_{dataset_name}_{version}"
    test_suite = giskard.Suite(name=suite_name)

    def calculate_rmse(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return rmse

    def rmse_test(model, dataset):
        y_true = dataset.df[dataset.target]
        y_pred = model.predict(dataset).raw
        rmse = calculate_rmse(y_true, y_pred)
        return giskard.TestResult(passed=rmse <= threshold)

    # Correctly adding the test to the suite
    test_suite.add_test(rmse_test, model=giskard_model, dataset=giskard_dataset, test_id="RMSE_Test_ID")
    
    test_results = test_suite.run()
    return test_results



def select_best_model(cfg, giskard_dataset, version, BASE_PATH):
    model_names = cfg.model.challenger_model_names
    model_aliases = ["challenger" + str(i+1) for i in range(len(model_names))]
    evaluation_metric_threshold = cfg.model.rmse_threshold

    client = mlflow.MlflowClient()
    best_model = None
    least_issues = float('inf')

    for model_name, model_alias in zip(model_names, model_aliases):
        # Assuming model_uri is constructed based on model_name and stored in a variable or fetched from a configuration
        model_uri = f"models:/{model_name}@{model_alias}"  # Example URI, adjust according to your MLflow setup
        feature_names = giskard_dataset.df.columns.tolist()  # Adjust based on actual feature names
        target_name = cfg.data.target_features[0]
        feature_names.remove(target_name)  # Ensure target_name is removed from feature_names
        
        giskard_model = load_and_wrap_mlflow_model(model_uri, feature_names, cfg)
        scan_results = run_giskard_scan(giskard_model, giskard_dataset, model_name, model_alias, giskard_dataset.name, BASE_PATH)
        test_results = create_and_run_test_suite(giskard_model, giskard_dataset, model_name, giskard_dataset.name, version, evaluation_metric_threshold)

        if test_results.passed:
            num_issues = len(scan_results.issues)
            if num_issues < least_issues:
                least_issues = num_issues
                best_model = (model_name, model_alias)

    return best_model


def find_model_version_by_alias(model_name, alias):
    client = mlflow.MlflowClient()
    # List all versions of the model
    model_versions = client.get_model_version_by_alias(model_name, alias)
    return model_versions.version


def tag_and_deploy_best_model(best_model):
    if best_model:
        model_name, model_alias = best_model
        client = mlflow.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            stage="Production",
            version=find_model_version_by_alias(model_name, model_alias)
        )
        print(f"Model {model_name} alias {model_alias} is deployed!!!")
    else:
        print("No valid model found. Please improve your models and try again.")


def main():
    BASE_PATH = os.getenv('PROJECTPATH')
    cfg = init_hydra()
    giskard_dataset, df, version = prepare_dataset(cfg)

    feature_names = giskard_dataset.df.columns.tolist()  # Assuming this is how you access the DataFrame and its columns
    
    # Remove the target column from feature names if it's included
    target_name = giskard_dataset.target
    feature_names.remove(target_name)

    best_model = select_best_model(cfg, giskard_dataset, version, BASE_PATH)
    tag_and_deploy_best_model(best_model)

if __name__ == "__main__":
    main()
