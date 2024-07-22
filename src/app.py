# src/app.py

import gradio as gr
import mlflow
from utils import init_hydra
from model import load_features
from transform_data import transform_data
import json
import requests
import numpy as np
import pandas as pd

cfg = init_hydra()

# You need to define a parameter for each column in your raw dataset
def predict(age = None,
            job = None,
            marital = None,
            education = None,
            default = None,
            balance = None,
            housing = None,
            loan = None,
            contact = None,
            day_of_week = None,
            month = None,
            duration = None,
            campaign = None,
            pdays = None,
            previous = None,
            poutcome = None):
    
    # This will be a dict of column values for input data sample
    features = {"age": age, 
        "balance": balance, 
        "duration": duration, 
        "campaign": campaign, 
        "pdays": pdays, 
        "previous": previous,
        "default": default, 
        "housing": housing, 
        "loan": loan,
        "day_of_week" : day_of_week,
        "month": month,
        "job": job,
        "marital": marital,
        "education": education,
        "contact": contact,
        "poutcome": poutcome
    }
    
    # print(features)
    
    # Build a dataframe of one row
    raw_df = pd.DataFrame(features, index=[0])
    
    # This will read the saved transformers "v4" from ZenML artifact store
    # And only transform the input data (no fit here).
    X = transform_data(
                        df = raw_df, 
                        cfg = cfg, 
                        return_df = False, 
                        only_transform = True, 
                        transformer_version = "v4", 
                        only_X = True
                      )
    
    # Convert it into JSON
    example = X.iloc[0,:]

    example = json.dumps( 
        { "inputs": example.to_dict() }
    )

    payload = example

    # Send POST request with the payload to the deployed Model API
    # Here you can pass the port number at runtime using Hydra
    response = requests.post(
        url=f"http://localhost:{port_number}/invocations",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    
    # Change this to some meaningful output for your model
    # For classification, it returns the predicted label
    # For regression, it returns the predicted value
    return response.json()

# Only one interface is enough
demo = gr.Interface(
    # The predict function will accept inputs as arguments and return output
    fn=predict,
    
    # Here, the arguments in `predict` function
    # will populated from the values of these input components
    inputs = [
        # Select proper components for data types of the columns in your raw dataset
        gr.Number(label = "age"), 
        gr.Text(label="job"),
        gr.Text(label="marital"),
        gr.Text(label="education"),
        gr.Dropdown(label="default", choices=["no", "yes"]),   
        gr.Number(label = "balance"), 
        gr.Dropdown(label="housing", choices=["no", "yes"]),   
        gr.Dropdown(label="loan", choices=["no", "yes"]),   
        gr.Text(label="contact"),
        gr.Text(label="day_of_week"),
        gr.Text(label="month"),
        gr.Number(label = "duration"), 
        gr.Number(label = "campaign"), 
        gr.Number(label = "pdays"), 
        gr.Number(label = "previous"),
        gr.Text(label="poutcome"),
    ],
    
    # The outputs here will get the returned value from `predict` function
    outputs = gr.Text(label="prediction result"),
    
    # This will provide the user with examples to test the API
    examples="data/examples"
    # data/examples is a folder contains a file `log.csv` 
    # which contains data samples as examples to enter by user 
    # when needed. 
)

# Launch the web UI locally on port 5155
demo.launch(server_port = 5155)
