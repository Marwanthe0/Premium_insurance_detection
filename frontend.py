# Importing necessary libraries
import streamlit as st
import requests

# Fetching the api key
API_URL = "http://127.0.0.1:8000/predict"

# Title and markdown of the project
st.title("Insurance premium category Predictor using Machine learning model")
st.markdown("Enter your details below")

# Input fields
age = st.number_input("Age", min_value=1, max_value=119, value=30)
weight = st.number_input("Weight", min_value=1.0, value=60.0)
height = st.number_input("Height", min_value=0.5, max_value=2.5, value=1.7)
income_lpa = st.number_input("Annual income in Lakhs(LPA)", min_value=0.0, value=8.0)
smoker = st.selectbox("Are you a smoker", options=[True, False])
city = st.text_input("City", value="Mumbai")
occcupation = st.selectbox(
    "Occupation",
    options=[
        "retired",
        "freelancer",
        "student",
        "government_job",
        "business_owner",
        "unemployed",
        "private_job",
    ]
)
