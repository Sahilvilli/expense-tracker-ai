"""
dashboard.py - Modern UI Edition (Final)
Author: Sahil Mandavilli (E23CSEU1474)
Clean, spacious, modern UI dashboard for Expense Tracking.
"""

import os
import sys
from datetime import datetime

import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px



# -------------------------------------
# Path Setup
# -------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

try:
    from predict import predict as predict_fn
except:
    predict_fn = None


# -------------------------------------
# Page & Theme Config
# -------------------------------------
st.set_page_config(
    page_title="Expense Tracker - Sahil Mandavilli",
    layout="wide",
)

# Modern UI Tweaks
st.markdown("""
<style>

section.main > div {
    padding-top: 1rem !important;
}

.card {
    background-color: #1f1f1f;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
    margin: 10px 0;
}

.stat-card {
    background-color: #1b1b1b;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 10px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.25);
}

</style>
""", unsafe_allow_html=True)


# -------------------------------------
# Load Data
# -------------------------------------
@st.cache_data
def load_data():
    path = os.path.join(DATA_DIR, "expenses.csv")
    return pd.read_csv(path, parse_dates=["date"])


def load_model_safe():
    preferred = os.path.join(MODELS_DIR, "rf_model.pkl")
    fallback = os.path.join(MODELS_DIR, "logistic_model.pkl")

    if os.path.exists(preferred):
        return joblib.load(preferred), preferred
    elif os.path.exists(fallback):
        return joblib.load(fallback), fallback
    else:
        raise FileNotFoundError("No trained model found.")


def safe_predict(amount, model):
    now = datetime.now()
    X = pd.DataFrame({
        "amount": [amount],
        "month": [now.month],
        "weekday": [now.weekday()]
    })
    try:
        return model.predict(X)[0]
    except:
        return model.predict(X.to_numpy())[0]


# -------------------------------------
# HEADER
# -------------------------------------
st.title("💸 Expense Tracker Dashboard")
st.write("### Modern UI • Powered by AI • Built by **Sahil Mandavilli (E23CSEU1474)**")
st.markdown("---")


# =====================================
# SIDEBAR — PREDICTION ONLY
# =====================================
with st.sidebar:
    st.markdown("## 🔮 Predict Category")

    amt = st.number_input("Amount", min_value=0.0, value=100.0)

    if st.button("Predict"):
        try:
            model, mpath = load_model_safe()
            if predict_fn:
                try:
                    pred = predict_fn(amt, model)
                except:
                    pred = safe_predict(amt, model)
            else:
                pred = safe_predict(amt, model)

            st.success(f"Predicted Category: **{pred}**")

        except Exception as e:
            st.error("Prediction failed.")
            st.write(e)

    st.markdown("---")
    st.caption("© Sahil Mandavilli")


# =====================================
# MAIN LAYOUT (Wide & Spacious)
# =====================================

df = load_data()
df["month"] = df["date"].dt.to_period("M")
df["weekday"] = df["date"].dt.day_name()

# TWO MAIN COLUMNS (spacious)
left, right = st.columns([2.2, 1.2])

# -------------------------------------
# LEFT SIDE — Charts
# -------------------------------------
with left:
    st.markdown("## 📊 Spending Overview")

    tab1, tab2 = st.tabs(["📈 Monthly Trend", "🥧 Category Breakdown"])

    # Monthly Line Chart
    with tab1:
        monthly = df.groupby(df["date"].dt.to_period("M"))["amount"].sum().reset_index()
        monthly["month"] = monthly["date"].dt.to_timestamp()

        st.plotly_chart(px.line(
            monthly,
            x="month",
            y="amount",
            markers=True,
            title="Monthly Spending Trend"
        ), use_container_width=True)

    # Pie Chart
    with tab2:
        category_totals = df.groupby("category")["amount"].sum().reset_index()
        fig = px.pie(
            category_totals,
            names="category",
            values="amount",
            hole=0.45,
            title="Spending by Category",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Sample Transactions
    st.markdown("## 📄 Sample Transactions")
    st.dataframe(df.head(50), use_container_width=True)

    # Category Count Table
    st.markdown("## 🧾 Category Counts")
    cat_counts = df["category"].value_counts().reset_index()
    cat_counts.columns = ["Category", "Count"]
    st.table(cat_counts)


# -------------------------------------
# RIGHT SIDE — Stats + Upload
# -------------------------------------
with right:
    st.markdown("## 📄 Dataset Stats")

    st.markdown(f"""
<div class="stat-card">
<b>Total Transactions:</b> {len(df)} <br>
<b>Total Spending:</b> ₹{df['amount'].sum():,.2f} <br>
<b>Unique Categories:</b> {df['category'].nunique()}
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 📤 Upload CSV")

    uploaded_file = st.file_uploader("Upload expenses CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            new_df = pd.read_csv(uploaded_file, parse_dates=["date"])
            st.success("CSV loaded!")
            st.dataframe(new_df.head(5))

            b1, b2 = st.columns(2)

            with b1:
                if st.button("💾 Replace Dataset"):
                    new_df.to_csv(os.path.join(DATA_DIR, "expenses.csv"), index=False)
                    st.cache_data.clear()
                    st.success("Dataset replaced!")
                    st.rerun()

            with b2:
                if st.button("🧠 Retrain Model"):
                    import importlib
                    tm = importlib.import_module("train_model")
                    tm.main()
                    st.success("Model retrained!")

        except Exception as e:
            st.error("Invalid CSV.")
            st.write(e)


# =====================================
# AI INSIGHTS SECTION
# =====================================
st.markdown("---")
st.markdown("## 🤖 AI Insights & Budget Suggestions")

monthly_sum = df.groupby(df["date"].dt.month)["amount"].sum()
top_month = int(monthly_sum.idxmax())
top_month_value = float(monthly_sum.max())

cat_sum = df.groupby("category")["amount"].sum()
top_cat = cat_sum.idxmax()
top_cat_value = float(cat_sum.max())

daily_avg = df.groupby("weekday")["amount"].mean().sort_values(ascending=False)

period = df.groupby(df["date"].dt.to_period("M"))["amount"].sum()
avg_month = float(period.mean())
latest_period = period.index.max()
latest_val = float(period.iloc[-1])
overspend = latest_val > avg_month * 1.2

# THREE INSIGHT CARDS
i1, i2, i3 = st.columns(3)

with i1:
    st.markdown(f"""
<div class="card">
<h3>📅 Highest Month</h3>
Month {top_month}<br>
<b>₹{top_month_value:,.2f}</b>
</div>
""", unsafe_allow_html=True)

with i2:
    st.markdown(f"""
<div class="card">
<h3>🏷 Top Category</h3>
{top_cat}<br>
<b>₹{top_cat_value:,.2f}</b>
</div>
""", unsafe_allow_html=True)

with i3:
    status = "⚠️ Overspending" if overspend else "✅ Normal"
    st.markdown(f"""
<div class="card">
<h3>📉 Status</h3>
{latest_period}<br>
<b>{status}</b>
</div>
""", unsafe_allow_html=True)

# Suggestions
st.markdown("### 💡 Smart Suggestions")

suggest = []
if top_cat_value > avg_month * 0.4:
    suggest.append(f"High spending detected in **{top_cat}** — consider category limits.")
if overspend:
    suggest.append("Reduce discretionary spending next month by 10–20%.")
if daily_avg.iloc[0] > df["amount"].mean() * 1.4:
    suggest.append(f"You overspend on **{daily_avg.index[0]}** — avoid impulse buys that day.")
if df["amount"].mean() > 500:
    suggest.append("Average purchase is high — consider cheaper alternatives.")
if not suggest:
    suggest.append("Your spending habits are healthy — great job!")

for s in suggest:
    st.write("•", s)
