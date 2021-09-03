import os
import sys

from flask import Flask
from flask_restful import Resource, Api, reqparse
from joblib import load
import numpy as np

MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_FILE = os.environ["MODEL_FILE"]
METADATA_FILE = os.environ["METADATA_FILE"]
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
METADATA_PATH = os.path.join(MODEL_DIR, METADATA_FILE)

print("Loading model from: {}".format(MODEL_PATH))
clf = load(MODEL_PATH)

app = Flask(__name__)
api = Api(app)


class Prediction(Resource):
    def __init__(self):
        self._required_features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
                                   'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B',
                                   'LSTAT']
        self.reqparse = reqparse.RequestParser()
        
        # Add features as arguments to flask_restful
        for feature in self._required_features:
            self.reqparse.add_argument(feature, 
                                       type = float, # Validate a float was passed
                                       required = True, # Requires the value to be presennt
                                       location = 'json', # Requires it to be passed as a json
                                       help = f'No {feature} provided' # Tells us which feature is missing
                                       )
        super(Prediction, self).__init__()

    def post(self):
        """
        First, we collect the incoming data by calling self.reqparse.parse_args(). 
        This automatically validates the request data according to the parameters 
        we passed to self.reqparse.add_argument in the init method.
        """
        # Values passed in POST request
        # args is a dictionary: {'CRIM': 15.02, 'ZN': 0.0, .... 'LSTAT': 24.9}
        args = self.reqparse.parse_args()
        print(f"Arg values passed: {args}")
        
        # Vectorize for input format
        X = np.array([args[f] for f in self._required_features]).reshape(1, -1)
        print(X)
        
        # Predict
        y_pred = clf.predict(X)
        
        return {'prediction': y_pred.tolist()[0]}

api.add_resource(Prediction, '/predict')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
