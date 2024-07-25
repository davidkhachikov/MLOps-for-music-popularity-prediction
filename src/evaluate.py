import argparse
import mlflow
import giskard

from mlflow.tracking import MlflowClient
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model import load_features
from utils import init_hydra

def evaluate(data_version="AIRFLOW2.0", model_name="hist_gradient_boosting", model_alias="champion"):
    client = MlflowClient()
    model_uri = f"models:/{model_name}@{model_alias}"
    model = mlflow.pyfunc.load_model(model_uri=model_uri)
    
    print(f"Evaluating model {model_uri} on data version {data_version}")

    X_test, y_test = load_features(name = "features_target", version=data_version)

    # Calculating prediction
    predictions = model.predict(X_test)

    # Calculating metrics
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)  # RMSE
    mae = mean_absolute_error(y_test, predictions)  # MAE

    r2 = r2_score(y_test, predictions)

    print(f"RMSE: {rmse}")
    print(f"R^2: {r2}")
    print(f"MAE: {mae}")

    # Loging results to mlflow
    with mlflow.start_run():
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model.")
    parser.add_argument("--data-version", type=str, default="AIRFLOW2.4", help="Data version to use for evaluation")
    parser.add_argument("--model-name", type=str, default="hist_gradient_boosting", help="Model name to evaluate")
    parser.add_argument("--model-alias", type=str, default="champion", help="Model alias to use for evaluation")
    args = parser.parse_args()

    evaluate(args.data_version, args.model_name, args.model_alias)
