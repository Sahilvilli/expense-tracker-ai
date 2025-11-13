# AI-Powered Personal Expense Tracker

**Author:** Sahil Mandavilli (E23CSEU1474)

This project implements an AI-based personal expense tracker that:
- Automatically categorizes expenses using ML models.
- Visualizes monthly spending trends.
- Shows basic suggestions for savings.

## Structure
```
expense-tracker-ai/
│── data/
│── models/
│── src/
│   ├── preprocess.py
│   ├── train_model.py
│   ├── predict.py
│── app/
│   ├── dashboard.py
│── requirements.txt
│── README.md
```

## Quick start
1. Create and activate a virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Put your dataset at `data/expenses.csv` (a synthetic example is included).
4. Train models: `python src/train_model.py`
5. Run the dashboard: `streamlit run app/dashboard.py`

