
"""
dashboard.py
Author: Sahil Mandavilli (E23CSEU1474)
Streamlit dashboard to interact with the expense tracker.
"""
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Expense Tracker - Sahil Mandavilli", layout="wide")
st.title("AI-Powered Personal Expense Tracker")
st.subheader("Author: Sahil Mandavilli (E23CSEU1474)")

@st.cache_data
def load_data():
    path = "data/expenses.csv"
    return pd.read_csv(path, parse_dates=["date"])


data = load_data()

st.sidebar.header("Predict a category")
amount = st.sidebar.number_input("Amount", min_value=0.0, value=100.0, step=10.0)
if st.sidebar.button("Predict (requires trained model)"):
    try:
        model = joblib.load(os.path.join("..","models","logistic_model.pkl"))
        import src.predict as predictor
        category = predictor.predict(amount, model=model)
        st.sidebar.success(f"Predicted category: {category}")
    except Exception as e:
        st.sidebar.error("Model not found or prediction failed. Train the model first using `python src/train_model.py`")
        st.sidebar.write(str(e))

st.header("Monthly spending overview")
data['month'] = data['date'].dt.to_period('M')
monthly = data.groupby('month')['amount'].sum().reset_index()
monthly['month'] = monthly['month'].dt.to_timestamp()
st.line_chart(monthly.rename(columns={'month':'index'}).set_index('index')['amount'])

st.header("Sample transactions")
st.dataframe(data.head(50))

st.header("Category distribution")
st.bar_chart(data['category'].value_counts())
