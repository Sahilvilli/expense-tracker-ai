"""
train_model.py
Upgraded training using RandomForest for stronger predictions.
Author: Sahil Mandavilli (E23CSEU1474)
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    return pd.read_csv(path, parse_dates=["date"])


# ---------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------
def get_X_y(df):
    df = df.copy()

    # Ensure required columns are present
    required = {"amount", "date", "category"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing required columns: {required - set(df.columns)}")

    # Extract time features
    df["month"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.weekday

    X = df[["amount", "month", "weekday"]]
    y = df["category"]
    return X, y


# ---------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------
def main():
    print("🔄 Loading data...")
    df = load_data("data/expenses.csv")

    print("🔄 Extracting features...")
    X, y = get_X_y(df)

    # Split data
    print("🔄 Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create model
    print("🧠 Training RandomForest model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    # Train
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    print(f"✅ Test accuracy: {score:.3f}")

    # Save model
    print("💾 Saving model...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/logistic_model.pkl")  # keep dashboard compatibility

    print("🎉 Model saved successfully at: models/logistic_model.pkl")


if __name__ == "__main__":
    main()
