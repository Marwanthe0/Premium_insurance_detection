import streamlit as st
import pandas as pd

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Insurance ML App",
    page_icon="🏠",
    layout="centered",
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 36px;
        font-weight: 800;
        color: #e5e7eb;
    }
    .subtitle {
        text-align: center;
        color: #9ca3af;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    .card {
        background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
        padding: 16px;
        border-radius: 14px;
        margin-bottom: 16px;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .metric-grid {
        display: flex;
        justify-content: space-between;
        margin-top: 10px;
    }
    .metric-box {
        flex: 1;
        text-align: center;
    }
    .metric-label {
        font-size: 13px;
        color: #9ca3af;
    }
    .metric-value {
        font-size: 22px;
        font-weight: 700;
        color: #e5e7eb;
        margin-top: 2px;
    }
    ul {
        padding-left: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="main-title">🏥 Insurance Premium Prediction System</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Machine Learning + FastAPI + Streamlit</div>',
    unsafe_allow_html=True,
)

st.divider()

# -----------------------------
# Project Overview
# -----------------------------
st.subheader("📌 Project Overview")

st.markdown(
    """
    <div class="card">
    This web application predicts <b>insurance premium categories</b> using 
    <b>Machine Learning models</b> based on user and vehicle attributes.

    <br><br>
    <b>Main Features:</b>
    <ul>
        <li>🏥 Health Insurance premium category prediction</li>
        <li>🚗 Car Insurance premium category prediction</li>
        <li>📊 Visual analytics for both datasets</li>
        <li>⚡ FastAPI-powered backend inference</li>
        <li>🎨 Streamlit-based interactive frontend</li>
    </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Load datasets
# -----------------------------
health_df = pd.read_csv("insurance.csv")
car_df = pd.read_csv("Car_Dataset.csv")

# -----------------------------
# Dataset Information
# -----------------------------
st.subheader("📂 Dataset Information")

# Health dataset card
st.markdown(
    f"""
    <div class="card">
        <h4>🏥 Health Insurance Dataset</h4>
        <div class="metric-grid">
            <div class="metric-box">
                <div class="metric-label">Records</div>
                <div class="metric-value">{health_df.shape[0]}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Features</div>
                <div class="metric-value">{health_df.shape[1]}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Target</div>
                <div class="metric-value">Categorical</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Car dataset card
st.markdown(
    f"""
    <div class="card">
        <h4>🚗 Car Insurance Dataset</h4>
        <div class="metric-grid">
            <div class="metric-box">
                <div class="metric-label">Records</div>
                <div class="metric-value">{car_df.shape[0]}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Features</div>
                <div class="metric-value">{car_df.shape[1]}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Target</div>
                <div class="metric-value">Categorical</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Navigation Hint
# -----------------------------
st.info(
    "👉 Use the sidebar to navigate between **Health Prediction**, **Car Prediction**, and **Analytics** pages."
)
