import argparse
import json
import requests
import numpy as np  # Import numpy for generating a random index
from model import load_features

def predict(version, docker_port, random_state):
    X, y = load_features(name="features_target",
                        version=version,
                        random_state=random_state
                        )

    # Generate a random index for selecting a random row
    random_index = np.random.randint(X.shape[0])

    example = X.iloc[random_index, :]
    example_target = y.iloc[random_index]

    example = json.dumps({"inputs": example.to_dict()})
    payload = example

    response = requests.post(
        url=f"http://localhost:{docker_port}/invocations",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    print(response.json())
    print("Randomly selected encoded target label: ", example_target)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict music popularity.')
    parser.add_argument('--version', type=str, required=True, help='Data version to use.')
    parser.add_argument('--port', type=int, required=True, help='Port number to communicate with the model.')
    parser.add_argument('--random_state', type=int, required=True, help='Random state for reproducibility.')

    args = parser.parse_args()

    predict(args.version, args.port, args.random_state)