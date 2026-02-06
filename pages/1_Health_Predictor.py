# frontend.py
import streamlit as st
import requests
from typing import Dict

# Page config & header styling (web-dev vibes)
st.set_page_config(
    page_title="Insurance Premium Predictor",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="auto",
)

st.markdown(
    """
<style>
.header {
  background: linear-gradient(90deg,#2563eb,#06b6d4);
  padding: 18px;
  border-radius: 12px;
  color: white;
  font-weight: 700;
  font-size: 20px;
  box-shadow: 0 6px 18px rgba(16,24,40,0.08);
}
.card {
  background: linear-gradient(180deg, rgba(255,255,255,0.85), rgba(255,255,255,0.7));
  padding: 16px;
  border-radius: 12px;
  box-shadow: 0 6px 18px rgba(2,6,23,0.06);
}
.small-muted { color: #6b7280; font-size: 13px; }
.result-badge { font-size:18px; font-weight:700; color:#111827; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="header">🏥 Insurance Premium Category Predictor — FastAPI + ML</div>',
    unsafe_allow_html=True,
)
st.write(
    "Enter your details below. The model runs on a FastAPI backend — make sure the API is running first. 🔌"
)

# default API URL (allow override)
API_URL = "http://127.0.0.1:8000/health/predict"


# quick-fill sample presets
preset = st.selectbox(
    "Quick sample",
    options=["— choose —", "Typical Adult", "Young Smoker", "Senior High Risk"],
)
if preset == "Typical Adult":
    default_values = {
        "age": 30,
        "weight": 70.0,
        "height": 5.6,
        "income_lpa": 6.0,
        "smoker": "No",
        "city": "Dhaka",
        "occupation": "private_job",
    }
elif preset == "Young Smoker":
    default_values = {
        "age": 22,
        "weight": 68.0,
        "height": 5.7,
        "income_lpa": 3.0,
        "smoker": "Yes",
        "city": "Dhaka",
        "occupation": "student",
    }
elif preset == "Senior High Risk":
    default_values = {
        "age": 62,
        "weight": 85.0,
        "height": 5.4,
        "income_lpa": 12.0,
        "smoker": "Yes",
        "city": "Dhaka",
        "occupation": "retired",
    }
else:
    default_values = {}

# layout columns for nicer form
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input(
            "Age", min_value=1, max_value=119, value=default_values.get("age", 30)
        )
        weight = st.number_input(
            "Weight (kg)", min_value=1.0, value=default_values.get("weight", 60.0)
        )
        height = st.number_input(
            "Height (feet)",
            min_value=3.0,
            max_value=8.0,
            value=default_values.get("height", 5.5),
            step=0.1,
        )
        income_lpa = st.number_input(
            "Annual income (LPA)",
            min_value=0.0,
            value=default_values.get("income_lpa", 8.0),
            step=0.1,
        )
    with col2:
        smoker = st.radio(
            "Smoker?",
            options=["No", "Yes"],
            index=0 if default_values.get("smoker", "No") == "No" else 1,
        )
        city = st.text_input("City", value=default_values.get("city", "Dhaka"))
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
            index=(
                0
                if default_values.get("occupation") is None
                else (
                    [
                        "retired",
                        "freelancer",
                        "student",
                        "government_job",
                        "business_owner",
                        "unemployed",
                        "private_job",
                        "Others",
                    ].index(default_values.get("occupation"))
                )
            ),
        )

    submit = st.form_submit_button("🔮 Predict Insurance Premium Category")


def build_payload() -> Dict:
    return {
        "age": int(age),
        "weight": float(weight),
        "height": float(height),
        "income_lpa": float(income_lpa),
        "smoker": True if smoker == "Yes" else False,
        "city": city.strip(),
        "occupation": occupation,
    }


# handle submit
if submit:
    payload = build_payload()

    # client-side basic validation
    if payload["income_lpa"] < 0:
        st.error("Income cannot be negative.")
    else:
        with st.spinner("Contacting model server..."):
            try:
                r = requests.post(API_URL, json=payload, timeout=8)
            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ Could not connect to FastAPI. Is it running? Start it with `uvicorn app:app --reload`."
                )
                st.stop()
            except requests.exceptions.Timeout:
                st.error(
                    "⏱️ Request timed out. Try again or increase the server timeout."
                )
                st.stop()

        if r.status_code == 200:
            try:
                data = r.json()
            except Exception:
                st.error("Invalid JSON response from server.")
                st.stop()

            # expected structure: {"response":{"predicted_category":"..."}}
            if "predicted_category" not in data:
                st.error("Unexpected API response format.")
                st.write(data)
            else:
                prediction = data["predicted_category"]
                confidence = data.get("confidence")

                st.success("✅ Predicted Insurance Premium Category")
                st.markdown(
                    f"""
                    <div class='card'>
                        <div class='result-badge'>🔹 {prediction}</div>
                        <div class='small-muted'>
                            Confidence: {round(confidence * 100, 2) if confidence else "N/A"}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:
            # show helpful error message if API returned a structured validation error
            try:
                err = r.json()
            except Exception:
                st.error(f"Server error ({r.status_code}).")
                st.write(r.text)
            else:
                st.error(f"Server returned {r.status_code}")
                # attempt to show pydantic details if present
                if isinstance(err, dict) and err.get("detail"):
                    st.write(err["detail"])
                else:
                    st.write(err)
