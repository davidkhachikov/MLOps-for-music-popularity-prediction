# Scripts

We utilize a variety of scripts to automate and facilitate our development and deployment processes. Below is a detailed explanation of each script and its purpose:

## Deploy Docker

- **Purpose**: The `deploy_docker` script is designed for pushing our machine learning model's container to Docker Hub. This script encapsulates the entire process of building a Docker image containing our ML model and then pushing it to a specified repository on Docker Hub.
- **Usage**: Run `./deploy_docker` in your terminal. This will build the Docker image and push it to [David Khachikov's Docker Hub repository](https://hub.docker.com/repository/docker/davidkhachikov/my_ml_service/general), where it will be available under the tag `general`. The container is configured to run the ML model on port 5152, making it accessible for inference purposes.

## Load to Remote

- **Purpose**: The `load_to_remote` script is tailored for use with Airflow DAGs. Its primary function is to ensure that the correct version of the data sample and ZenML artifacts are pushed to the remote server. This script simplifies the process of managing versions and dependencies, ensuring that our deployments are consistent and reproducible.
- **Usage**: Execute `./load_to_remote` to initiate the process. This script will automatically handle the selection and transfer of the appropriate data and artifacts, streamlining the deployment process.

## Predict Samples

- **Purpose**: The `predict_samples` script leverages MLflow to predict outcomes using our champion model. It is designed to accept various datasets, allowing for flexible testing and evaluation of our model against different inputs.
- **Usage**: Run `./predict_samples` followed by any necessary arguments to specify the dataset(s) you wish to use for prediction. This script utilizes the MLflow entry point to execute predictions, providing insights into the performance of our model.

## Test API

- **Purpose**: The `test_api` script is dedicated to testing the functionality of our Flask application. It performs automated tests on the application, ensuring that it operates correctly and efficiently.
- **Usage**: Simply execute `./test_api` to start the test suite. This script will simulate requests to the Flask app and verify that responses are as expected, helping to identify and resolve any issues before deployment.

## Test Data

- **Note**: Previously known as the `old_script`, the `test_data` script was initially designed to perform a similar function to the current Airflow `data_extract` DAG. While it has been superseded by more advanced automation, it remains a valuable resource for understanding past data processing methodologies.
- **Usage**: For historical or educational purposes, `./test_data` can still be executed. However, it is recommended to use the newer Airflow DAG for data extraction tasks moving forward.

These scripts form the backbone of our development and deployment processes, automating critical tasks and ensuring the smooth operation of our ML services.