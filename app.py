# app.py
import logging
import pickle
from typing import Literal, Annotated, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, computed_field
import pandas as pd

logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="Insurance Premium Predictor API",
    description="Predict insurance premium category from user features.",
    version="1.0.0",
)

# Allow requests from Streamlit and local dev tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",  # streamlit default
        "http://127.0.0.1:8501",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# global model holder
model = None
model_loaded = False
sklearn_version: Optional[str] = None


@app.on_event("startup")
def load_model_on_startup():
    global model, model_loaded, sklearn_version
    try:
        # optional: log installed sklearn version
        try:
            import sklearn

            sklearn_version = sklearn.__version__
            logger.info("scikit-learn installed: %s", sklearn_version)
        except Exception:
            sklearn_version = None
            logger.warning("Could not determine installed scikit-learn version.")

        # load the pickled model (wrap errors)
        with open("models/model.pkl", "rb") as f:
            model = pickle.load(f)

        model_loaded = True
        logger.info("Model loaded successfully.")
    except Exception as e:
        model = None
        model_loaded = False
        logger.exception("Failed to load model at startup: %s", e)


# City tiers (kept from your original data)
tier_1 = [...]
tier_2 = [...]


class UserInput(BaseModel):
    age: Annotated[int, Field(..., gt=0, lt=120)]
    weight: Annotated[float, Field(..., gt=0)]
    height: Annotated[float, Field(..., gt=0)]
    income_lpa: Annotated[float, Field(..., gt=0)]
    smoker: Annotated[bool, Field(...)]
    city: Annotated[str, Field(...)]
    occupation: Annotated[
        Literal[
            "retired",
            "freelancer",
            "student",
            "government_job",
            "business_owner",
            "unemployed",
            "private_job",
            "Others",
        ],
        Field(...),
    ]

    @computed_field
    @property
    def bmi(self) -> float:
        # height is given in feet; convert to meters
        return self.weight / ((self.height * 0.3048) ** 2)

    @computed_field
    @property
    def lifestyle_risk(self) -> str:
        if self.smoker and self.bmi > 30:
            return "high"
        elif self.smoker and self.bmi > 27:
            return "medium"
        else:
            return "low"

    @computed_field
    @property
    def age_group(self) -> str:
        if self.age < 25:
            return "Young"
        elif self.age < 45:
            return "Adult"
        elif self.age < 60:
            return "Middle_Aged"
        else:
            return "Senior"

    @computed_field
    @property
    def city_tier(self) -> int:
        if self.city in tier_1:
            return 1
        elif self.city in tier_2:
            return 2
        else:
            return 3


@app.get("/health")
def health():
    """Simple health check to verify the model and environment."""
    return {
        "status": "ok" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded,
        "sklearn_version": sklearn_version,
    }


@app.post("/predict")
def predict_premium(data: UserInput):
    """Predict endpoint. Returns 503 if model failed to load at startup."""
    if not model_loaded or model is None:
        logger.error("Prediction requested but model not loaded.")
        raise HTTPException(
            status_code=503, detail="Model not loaded. Check server logs."
        )

    # build dataframe matching training feature layout
    input_df = pd.DataFrame(
        [
            {
                "bmi": data.bmi,
                "age_group": data.age_group,
                "lifestyle_risk": data.lifestyle_risk,
                "city_tier": data.city_tier,
                "income_lpa": data.income_lpa,
                "occupation": data.occupation,
            }
        ]
    )

    try:
        prediction = model.predict(input_df)[0]
    except Exception as e:
        logger.exception("Error during model prediction: %s", e)
        raise HTTPException(status_code=500, detail="Error during model prediction.")

    return JSONResponse(
        status_code=200, content={"response": {"predicted_category": str(prediction)}}
    )
