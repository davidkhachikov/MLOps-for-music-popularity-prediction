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
def predict(artist_followers=None,
            genres=None,
            album_total_tracks=None,
            artist_popularity=None,
            explicit=None,
            tempo=None,
            chart=None,
            album_release_date=None,
            energy=None,
            key=None,
            popularity=None,
            available_markets=None,
            mode=None,
            time_signature=None,
            album_name=None,
            speechiness=None,
            danceability=None,
            valence=None,
            acousticness=None,
            liveness=None,
            instrumentalness=None,
            loudness=None,
            name=None):
    
    # This will be a dict of column values for input data sample
    features = {
        "artist_followers": artist_followers,
        "genres": genres,
        "album_total_tracks": album_total_tracks,
        "artist_popularity": artist_popularity,
        "explicit": explicit,
        "tempo": tempo,
        "chart": chart,
        "album_release_date": album_release_date,
        "energy": energy,
        "key": key,
        "popularity": popularity,
        "available_markets": available_markets,
        "mode": mode,
        "time_signature": time_signature,
        "album_name": album_name,
        "speechiness": speechiness,
        "danceability": danceability,
        "valence": valence,
        "acousticness": acousticness,
        "liveness": liveness,
        "instrumentalness": instrumentalness,
        "loudness": loudness,
        "name": name
    }
    
    # print(features)
    
    # Build a dataframe of one row
    raw_df = pd.DataFrame(features, index=[0])
    cfg = init_hydra()

    # Drop unnecessary columns
    raw_df.drop(columns=cfg.data.low_features_number, inplace=True, errors='ignore')
    # preprocess datetime features
    for feature in cfg.data.timedate_features:
        raw_df[feature] = raw_df[feature].apply(lambda d: pd.Timestamp(d) if pd.notnull(d) and d != '' else pd.Timestamp("1970-01-01"))

    for feature in cfg.data.missing_list:
        raw_df[feature] = raw_df[feature].apply(lambda d: d if pd.notnull(d) and d != '' else '[]')

    for feature in cfg.data.missing_strings:
        raw_df[feature] = raw_df[feature].apply(lambda d: d if pd.notnull(d) and d != '' else ' ')
        
    # Binarize categorical features
    raw_df["chart"] = raw_df["chart"].map({"top200": 1, "top50": 2})
    raw_df["chart"] = raw_df["chart"].fillna(0)

    # Impute missing values with median
    raw_df.fillna(raw_df.median(), inplace=True)
    raw_df.to_csv('zenml_input.csv')
    X = transform_data(
                        df = raw_df, 
                        cfg = cfg, 
                        return_df = False, 
                        only_transform = True, 
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
    fn=predict,
    inputs=[
        gr.Number(label="artist_followers"),
        gr.Text(label="genres"),
        gr.Number(label="album_total_tracks"),
        gr.Number(label="artist_popularity"),
        gr.Checkbox(label="explicit"),
        gr.Number(label="tempo"),
        gr.Text(label="chart"),
        gr.Text(label="album_release_date"),
        gr.Number(label="energy"),
        gr.Number(label="key"),
        gr.Number(label="popularity"),
        gr.Text(label="available_markets"),
        gr.Number(label="mode"),
        gr.Number(label="time_signature"),
        gr.Text(label="album_name"),
        gr.Number(label="speechiness"),
        gr.Number(label="danceability"),
        gr.Number(label="valence"),
        gr.Number(label="acousticness"),
        gr.Number(label="liveness"),
        gr.Number(label="instrumentalness"),
        gr.Number(label="loudness"),
        gr.Text(label="name")
    ],
    outputs=gr.Text(label="prediction result"),
    # examples="data/examples"
)

# Launch the web UI locally on port 5155
demo.launch(server_port = cfg.web_ui_port)
