## Api

This folder contains the necessary files and scripts to set up and deploy the API for our music popularity prediction model. The API is built using Flask and leverages MLflow for model management.
Files and Directories
api/app.py

This is the main application file for the API. It contains the following endpoints:

    /info (GET): Provides information about the deployed model.
    / (GET): A welcome message describing the API and its endpoints.
    /predict (POST): Endpoint for sending prediction requests to the deployed model.