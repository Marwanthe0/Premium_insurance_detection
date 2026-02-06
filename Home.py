import streamlit as st
import pandas as pd

st.set_page_config(page_title="Insurance ML App", page_icon="🏠", layout="centered")

st.markdown(
    """
    <h1 style="text-align:center;">🏥 Insurance Premium Prediction System</h1>
    <p style="text-align:center; color:gray;">
    Machine Learning + FastAPI + Streamlit
    </p>
    """,
    unsafe_allow_html=True,
)

st.divider()

st.subheader("📌 Project Overview")
st.write(
    """
This web application predicts **insurance premium categories** based on
user lifestyle, demographics, and income using a **Machine Learning model**.

The system consists of:
- **FastAPI** → Backend ML inference
- **Scikit-learn** → Trained ML model
- **Streamlit** → Interactive web frontend
"""
)

st.subheader("📂 Dataset Information")
df = pd.read_csv("insurance.csv")

col1, col2, col3 = st.columns(3)
col1.metric("Total Records", df.shape[0])
col2.metric("Total Features", df.shape[1])
col3.metric("Target Type", "Categorical")

st.info("Navigate using the sidebar 👈 to explore analytics or predict premiums.")
