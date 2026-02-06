# pages/3_Car_Predict.py
import streamlit as st
import requests
from typing import Dict

st.set_page_config(
    page_title="Car Insurance Category Predict", page_icon="🚗", layout="centered"
)

st.title("🚗 Car Insurance — Predict (dataset fields only)")
st.write("This form asks only the exact features the Car model was trained on.")

API_URL = st.text_input("API URL", value="http://127.0.0.1:8000/car/predict")

st.info(
    "Enter annual mileage in *thousands* of km. e.g., 120 means 120,000 km (so input 120.0)."
)

with st.form("car_form"):
    c1, c2 = st.columns(2)
    with c1:
        driver_age = st.number_input(
            "Driver Age (years)", min_value=16, max_value=120, value=30
        )
        driver_experience = st.number_input(
            "Driver Experience (years)", min_value=0, max_value=100, value=5
        )
        previous_accidents = st.number_input(
            "Previous Accidents (count)", min_value=0, max_value=50, value=0
        )
    with c2:
        annual_mileage_x1000 = st.number_input(
            "Annual Mileage (x1000 km)", min_value=0.0, value=12.0, step=0.1
        )
        car_manufacturing_year = st.number_input(
            "Car Manufacturing Year", min_value=1900, max_value=2100, value=2015
        )
        car_age = st.number_input(
            "Car Age (years)", min_value=0, max_value=100, value=6
        )
    submit = st.form_submit_button("🔮 Predict Car Premium Category")


def build_payload() -> Dict:
    return {
        "driver_age": int(driver_age),
        "driver_experience": int(driver_experience),
        "previous_accidents": int(previous_accidents),
        "annual_mileage_x1000": float(annual_mileage_x1000),
        "car_manufacturing_year": int(car_manufacturing_year),
        "car_age": int(car_age),
    }


if submit:
    payload = build_payload()
    st.write("Sending payload to API:")
    st.json(payload)

    if API_URL.strip() == "":
        st.error("Set the API URL first.")
    else:
        with st.spinner("Contacting prediction server..."):
            try:
                res = requests.post(API_URL, json=payload, timeout=8)
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to FastAPI. Is it running?")
                st.stop()
            except requests.exceptions.Timeout:
                st.error("Request timed out.")
                st.stop()

        if res.status_code != 200:
            st.error(f"Server returned {res.status_code}")
            try:
                st.write(res.json())
            except Exception:
                st.write(res.text)
        else:
            data = res.json()
            pred = data.get("predicted_category")
            conf = data.get("confidence")
            if pred is None:
                st.error("Unexpected response format.")
                st.write(data)
            else:
                st.success(f"✅ Predicted Category: {pred}")
                if conf is not None:
                    st.info(f"Confidence: {round(float(conf) * 100, 2)}%")
                advice = {
                    "Low": "Low risk — standard premium expected.",
                    "Medium": "Medium risk — consider inspection.",
                    "High": "High risk — higher premium likely.",
                }
                st.write(advice.get(pred, "Interpret result carefully."))
