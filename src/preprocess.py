"""
preprocess.py
Author: Sahil Mandavilli (E23CSEU1474)
Simple data loading and preprocessing utilities for the expense tracker.
"""
import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def basic_features(df):
    # create useful features: day, month, year, weekday
    df = df.copy()
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['weekday'] = df['date'].dt.weekday
    # fill missing category with 'unknown'
    df['category'] = df['category'].fillna('unknown')
    return df

def get_X_y(df):
    df2 = basic_features(df)
    # For a simple baseline, use amount + month + weekday as features
    X = df2[['amount', 'month', 'weekday']]
    y = df2['category']
    return X, y

if __name__ == "__main__":
    df = load_data("data/expenses.csv")
    X, y = get_X_y(df)
    print("Loaded data:", df.shape)
    print("X shape:", X.shape, "y shape:", y.shape)
