# api/app.py

from flask import Flask, request, jsonify, abort, make_response

import mlflow
import mlflow.pyfunc
import os

BASE_PATH = os.path.expandvars("$PROJECTPATH")

model = mlflow.pyfunc.load_model(os.path.join(BASE_PATH, "api", "model_dir"))

app = Flask(__name__)

@app.route("/info", methods = ["GET"])
def info():

	response = make_response(str(model.metadata), 200)
	response.content_type = "text/plain"
	return response

@app.route("/", methods = ["GET"])
def home():
	msg = """
	Welcome to our ML service to predict Customer satisfaction\n\n

	This API has two main endpoints:\n
	1. /info: to get info about the deployed model.\n
	2. /predict: to send predict requests to our deployed model.\n

	"""

	response = make_response(msg, 200)
	response.content_type = "text/plain"
	return response

# /predict endpoint
@app.route("/predict", methods = ["POST"])
def predict():
	
    # EDIT THIS ENDPOINT
    
    # EXAMPLE
	content = str(request.data)
	response = make_response(content, 200)
	response.headers["content-type"] = "application/json"
	return jsonify({'result': 'yes', 'prob': '0.7'})

# This will run a local server to accept requests to the API.
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)