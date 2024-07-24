# api/app.py

import logging

# Add this line near the top of your file, after other imports
logger = logging.getLogger(__name__)


from flask import Flask, request, jsonify, abort, make_response
from model import model_predict

import mlflow
import mlflow.pyfunc
import os

import pandas as pd

from utils import init_hydra

BASE_PATH = os.path.expandvars("$PROJECTPATH")
model_path = f'{BASE_PATH}/models/champion'
model = mlflow.pyfunc.load_model(model_path)
app = Flask(__name__)

@app.route("/info", methods = ["GET"])
def info():
	
	response = make_response(str(model.metadata), 200)
	response.content_type = "text/plain"
	return response

@app.route("/predict", methods=["POST"])
def predict():
    data = None
    try:
        # Load the JSON data from the request
        data = request.json
        logger.info(f"Received data: {data}")
        
        # Validate input
        if 'inputs' not in data:
            raise ValueError("Missing 'inputs' in request data")
        
        inputs = data['inputs']
        
        # Convert the dictionary to a DataFrame for easier manipulation
        df = pd.DataFrame([inputs])
        
        # Make a prediction
        prediction = model_predict(df, model)
        
        # Assuming the model returns a numpy array, convert it to a list for JSON serialization
        prediction_list = prediction.tolist()
        
        # Format the prediction result
        result = {
            'result': 'success',
            'prediction': prediction_list
        }
        
        return jsonify(result), 200
    except Exception as e:
        # Log the data that caused the failure
        logger.error(f"Error processing data: {data}, Error: {str(e)}")
        # Send the logged data as part of the error response
        return jsonify({'error': str(e), 'logged_data': data}), 400

# This will run a local server to accept requests to the API.
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)