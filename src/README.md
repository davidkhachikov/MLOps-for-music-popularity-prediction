# Source code

## `data.py`: Data Handling Module

The `data.py` script plays a crucial role in the  project by handling various aspects of data management, preprocessing, and feature engineering. It is designed to streamline the process of preparing data for machine learning models, ensuring that the data is clean, structured, and optimized for predictive modeling. Below is a breakdown of its key functionalities:

### Data Acquisition

- **Sampling Data**: Utilizes Hydra configurations to sample a subset of the music popularity dataset from a CSV file. This operation ensures that only a manageable portion of the data is processed at any given time, reducing computational overhead.
- **Initial Data Handling**: Cleans and transforms the raw dataset by converting relevant columns to datetime objects, handling missing values, and performing initial exploratory data transformations such as binarizing categorical features.

### Data Preprocessing

- **Word Vectorization**: Implements functions to average word vectors for text data, facilitating the conversion of textual information into numerical features that can be understood by machine learning algorithms.
- **Feature Engineering**: Applies various transformations to the dataset, including fitting and saving transformers for uniform, normal, one-hot encoded, and multilabel features. It also handles date features by creating cyclical features and normalizes text features using Word2Vec models.
- **Data Transformation**: Handles different types of features (uniform, normal, one-hot encoded, etc.) through specialized functions that apply appropriate preprocessing techniques, such as scaling, imputation, and encoding.

### Feature Validation

- **Expectation Suite Validation**: Uses Great Expectations to validate the features against predefined expectations, ensuring data quality and consistency across different versions of the dataset.

### Feature Loading

- **Artifact Management**: Saves and loads features and targets as ZenML artifacts, allowing for version-controlled data management and easy retrieval of processed data.

Overall, `data.py` serves as a central hub for all data-related operations within the project, ensuring that the data is properly prepared and validated before being fed into machine learning models for training and inference.


## `predict.py`

The `predict.py` script is designed to facilitate predictions using the trained machine learning models within the MLOps for Music Popularity Prediction project. It leverages the Hydra framework for configuration management and integrates seamlessly with the project's data processing pipeline to make predictions on new data instances. Here's a concise overview of its functionality:

## Description of `transform_data.py` and function inside

The `transform_data` function is a pivotal component within the project, specifically designed to preprocess and prepare data for machine learning models. It orchestrates the transformation and validation of input data according to predefined configurations and options, ensuring that the data is appropriately formatted and cleaned before being utilized for model training or inference.

## Description of `evaluate.py`

The `evaluate.py` script is designed to assess the performance of machine learning models within the project by calculating and logging key metrics. It leverages MLflow for model evaluation and tracking, ensuring a systematic approach to model assessment and comparison. Here's a concise overview:

- **Model Evaluation**: Loads a specified model from MLflow Model Registry using its name and alias, then evaluates it against a given data version.
- **Metrics Calculation**: Computes RMSE, MAE, and R^2 score for the model's predictions, providing insights into its predictive accuracy.
- **MLflow Integration**: Logs evaluation metrics (RMSE and R^2) to MLflow, facilitating model performance tracking and comparison.
- **Command-Line Interface**: Utilizes `argparse` for command-line arguments, allowing users to specify data version, model name, and model alias, enhancing flexibility and usability.

## Short Description of `src/app.py`

The `app.py` script is a core component of the project, designed to facilitate model predictions through a **cool** user-friendly interface. It leverages Gradio for creating an interactive web UI, allowing users to input music track features and receive predictions on track popularity. Here's a concise overview:

- **Data Preparation**: Constructs a DataFrame from user inputs, preprocesses it according to project configurations, and transforms it for model compatibility.
- **Model Invocation**: Sends transformed data to a deployed model via a POST request, showcasing integration with model deployment systems.
- **Gradio Interface**: Provides an intuitive UI for data input and displays model predictions, enhancing user interaction and accessibility.
- **Configuration Management**: Utilizes Hydra for managing configurations, ensuring flexibility and ease of use.

## Overview of `src/validate.py`

The `src/validate.py` script focuses on model validation and testing. It integrates with Giskard for model validation, MLflow for model management, and utilizes Hydra for configuration. Here's a detailed breakdown of its functionalities:

### Process Flow:

1. **Dataset Preparation**: Loads and prepares the dataset based on the configuration, ensuring compatibility with Giskard's validation framework.
2. **Model Loading**: Supports loading models from both pickle files and MLflow, wrapping them for validation.
3. **Validation Execution**: Runs a suite of tests, primarily focusing on MAE, and generates detailed reports.
4. **Model Version Management**: Facilitates version control by retrieving model versions from MLflow.
5. **Main Workflow**: Initializes configurations, prepares datasets, and performs model validation, ensuring models meet quality standards before deployment.

### Usage:

- Validates models against predefined metrics, ensuring reliability and performance standards are met.
- Generates detailed validation reports for further analysis and model comparison.
- Integrates with MLflow for model versioning and management, enhancing model lifecycle management.
