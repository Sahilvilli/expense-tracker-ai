"""
predict.py
Author: Sahil Mandavilli (E23CSEU1474)
Load saved model and predict category for a given transaction.
"""
import joblib
import pandas as pd

def load_model(path="../models/logistic_model.pkl"):
    return joblib.load(path)

def predict(amount, model=None):
    if model is None:
        model = load_model()
    df = pd.DataFrame({'amount':[amount], 'month':[1], 'weekday':[0]})
    return model.predict(df)[0]

if __name__ == "__main__":
    print("Example prediction (amount=250):")
    try:
        print(predict(250))
    except Exception as e:
        print("Prediction failed (model may not be trained). Error:", e)
