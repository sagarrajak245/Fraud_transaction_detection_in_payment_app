import joblib
import numpy as np

# Load the model from the .pkl file
model = joblib.load("xgb_fraud_detection_model.pkl")

def predict(transaction: dict):
    features = [
        transaction["amount"],
        transaction["oldbalanceOrg"],
        transaction["newbalanceOrig"],
        transaction["oldbalanceDest"],
        transaction["newbalanceDest"],
        transaction["errorBalanceOrig"],
        transaction["errorBalanceDest"],
        transaction["type_CASH_OUT"],
        transaction["type_TRANSFER"]
    ]
    
    prediction = model.predict([features])[0]
    proba = model.predict_proba([features])[0][1]

    return {
        "prediction": int(prediction),
        "probability_of_fraud": round(float(proba), 4)
    }
