"""
train_model.py
Author: Sahil Mandavilli (E23CSEU1474)
Train a baseline Logistic Regression and save it to models/.
"""
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from src.preprocess import load_data, get_X_y

def main():
    df = load_data("data/expenses.csv")
    X, y = get_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("Test accuracy:", acc)
    print(classification_report(y_test, preds))
    joblib.dump(model, "models/logistic_model.pkl")
    print("Saved model to ../models/logistic_model.pkl")

if __name__ == "__main__":
    main()
