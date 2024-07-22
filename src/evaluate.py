import argparse
import mlflow
import giskard

from mlflow.tracking import MlflowClient
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from model import load_features
from utils import init_hydra

def evaluate(cfg, model_name, model_alias="champion"):
    client = MlflowClient()
    model_uri = f"models:/{model_name}@{model_alias}"
    model = mlflow.pyfunc.load_model(model_uri=model_uri)
    
    data_version = cfg.data.version
    print(f"Evaluating model {model_uri} on data version {data_version}")

    X_test, y_test = load_features(name = "features_target", version=data_version)

    # Вычисление предсказаний модели
    predictions = model.predict(X_test)

    # Calculating metrics
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)  # RMSE
    r2 = r2_score(y_test, predictions)

    print(f"RMSE: {rmse}")
    print(f"R^2: {r2}")

    # Логирование результатов в MLflow
    with mlflow.start_run():
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model.")
    parser.add_argument("--model-name", type=str, default="neural_network", help="Model name to evaluate")
    args = parser.parse_args()

    evaluate(init_hydra(), args.model_name)