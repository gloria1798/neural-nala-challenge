"""Flask web server serving text_recognizer predictions."""
import os
import logging
from flask import Flask, request, jsonify
from analytics.src.data import FraudData
from analytics.src.models import RandomForest
import analytics.util as util

import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

fraud_data_obj = FraudData()

app = Flask(__name__)  # pylint: disable=invalid-name
model = RandomForest()

logging.basicConfig(level=logging.INFO)

@app.route("/")
def index():
    """Provide simple health check route."""
    return "Hello, world!"

@app.route("/v1/predict", methods=["GET", "POST"])
def predict():
	"""Provide main prediction API route. Responds to both GET and POST requests."""       
	data = _load_data()
	data = FraudData.clean_data_query(data)
	print("after clean query: ", data)
	data = fraud_data_obj.data_preprocess(data)
	pred = model.predict(data)
	return jsonify({"pred": str(pred)})

def _load_data():
    if request.method == "POST":
        data = request.get_json()
        if data is None:
            return "No json received"       
        return data
    if request.method == "GET":
        data_url = request.args.get("data_url")
        if data_url is None:
            return "no url defined in query string"        
        return "get is pressed"
    raise ValueError("Unsupported HTTP method")


def main():
    """Run the app."""
    app.run(host="0.0.0.0", port=8000, debug=False)  # nosec


if __name__ == "__main__":
    main()
