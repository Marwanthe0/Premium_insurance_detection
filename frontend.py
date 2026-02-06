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
height = st.number_input("Height (In Feet)", min_value=3.5, max_value=7, value=5.5)
income_lpa = st.number_input("Annual income in Lakhs(LPA)", min_value=0.0, value=8.0)
smoker = st.selectbox("Are you a smoker", options=[True, False])
city = st.text_input("City", value="Dhaka")
occupation = st.selectbox(
    "Occupation",
    options=[
        "retired",
        "freelancer",
        "student",
        "government_job",
        "business_owner",
        "unemployed",
        "private_job",
        "Others",
    ],
)

if st.button("Predict Insurance Premium Category"):
    input_data = {
        "age": age,
        "weight": weight,
        "height": height,
        "income_lpa": income_lpa,
        "smoker": smoker,
        "city": city,
        "occupation": occupation,
    }

    try:
        response = requests.post(API_URL, json=input_data)
        result = response.json()

        if response.status_code == 200 and "response" in result:
            prediction = result["response"]
            st.success(
                f"Predicted INsurance Premium Category: **{prediction['predicted_category']}**"
            )
        else:
            code = response.status_code
            st.error(f"Error: {code}")
            st.write(
                "Error occured:",
                result["detail"][0]["loc"][1],
                "-",
                result["detail"][0]["msg"],
            )
    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to the FastAPI server. Make sure it's running.")
