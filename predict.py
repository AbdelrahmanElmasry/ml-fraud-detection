from flask import Flask
from flask import request
from flask import jsonify 

import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler
import xgboost as xgb
import numpy as np
import requests

app = Flask('fraud-detection-model')
numeric_columns = ['step','type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

loaded_model = xgb.Booster()
loaded_model.load_model('model_fraud.model')

def normalize_input(x):
    X = pd.DataFrame().from_dict(x)
    X['isC2C'] = np.array(x['nameOrig'][0].startswith('C') & x['nameDest'][0].startswith('C')).astype(int)
    X.drop(['nameOrig', 'nameDest'],axis=1, inplace=True)

    return X

def preprocess_input(X):
    label_encoder = LabelEncoder()
    scaler = RobustScaler()


    X['type'] = label_encoder.fit_transform(X['type'])

    for column in numeric_columns:
        X[column] = np.log1p(X[column])
    scaler.fit(X[numeric_columns])
    # X[numeric_columns] = scaler.transform(X[numeric_columns])

    ## Resort to match the same model features order
    desired_order = ['amount', 'isC2C', 'newbalanceDest', 'newbalanceOrig', 'oldbalanceDest', 'oldbalanceOrg', 'step', 'type']
    X = X.reindex(columns=desired_order)

    return X


@app.route('/predict', methods=['POST'])
def predict():
    transaction = request.get_json()

    X = normalize_input(transaction)
    processed_X = preprocess_input(X)


    test_data = xgb.DMatrix(X)
    fraudProba = np.expm1(loaded_model.predict(test_data)[0])
    predictionIsFraud = ( fraudProba >= 0.5) == 1

    result = {
        # 'fraud_probability': float(fraudProba),
        'isFraud': bool(predictionIsFraud)
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)