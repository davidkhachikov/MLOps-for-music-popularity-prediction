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
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Load the JSON data from the request
        data = request.json
        # Extract the 'inputs' dictionary from the JSON
        inputs = data['inputs']
        
        # Convert the dictionary to a DataFrame for easier manipulation
        df = pd.DataFrame(inputs, index=[0])
        
        # Preprocess the data if necessary. This step depends on how your model was trained.
        # For example, if your model expects numerical columns to be scaled, apply the same scaling here.
        
        # Make a prediction
        # Note: Ensure your model can handle a DataFrame of shape (1, num_features) where num_features matches the model's expectation
        prediction = model.predict(df)
        
        # Assuming the model returns a numpy array, convert it to a list for JSON serialization
        prediction_list = prediction.tolist()
        
        # Format the prediction result
        result = {
            'result': 'success',
            'prediction': prediction_list
        }
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}, 400)

# This will run a local server to accept requests to the API.
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)