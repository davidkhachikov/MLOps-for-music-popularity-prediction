## Api

This folder contains the necessary files and scripts to set up and deploy the API for our music popularity prediction model. The API is built using Flask and leverages MLflow for model management.

### Files and Directories

`api/app.py`

This is the main application file for the API. It contains the following endpoints:

    /info (GET): Provides information about the deployed model.
    / (GET): A welcome message describing the API and its endpoints.
    /predict (POST): Endpoint for sending prediction requests to the deployed model.

You can use `scripts/test_api.sh` to test api/predict endpoint on example. You can find examples in api/examples folder. Also you can run predict endpoint of mlflow to try model on random element of sample.

`scripts/deploy_docker.sh`

This script is used to build and run the Docker container for the API.

`Dockerfile`

This Dockerfile is designed for deploying the API using Docker. It sets up the environment, installs necessary dependencies, and configures the MLflow model server.