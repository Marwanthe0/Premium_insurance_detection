import logging
import os
import joblib
from typing import Literal, Annotated, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, computed_field
import pandas as pd

logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="Insurance Premium Predictor API",
    description="Health & Car insurance premium category prediction API",
    version="1.0.0",
)

# allow local frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

models = {"health": None, "car": None}
models_loaded = {"health": False, "car": False}
sklearn_version: Optional[str] = None


# Model Loading
@app.on_event("startup")
def load_models():
    global sklearn_version
    try:
        import sklearn

        sklearn_version = sklearn.__version__
        logger.info("scikit-learn version: %s", sklearn_version)
    except Exception:
        sklearn_version = None

    # health model (optional)
    health_path = "models/health_insurance_model.pkl"
    if os.path.exists(health_path):
        try:
            models["health"] = joblib.load(health_path)
            models_loaded["health"] = True
            logger.info("Loaded health model from %s", health_path)
        except Exception as e:
            models["health"] = None
            models_loaded["health"] = False
            logger.exception("Failed to load health model: %s", e)
    else:
        logger.warning("Health model not found at %s", health_path)

    # car model
    car_path = "models/car_insurance_model.pkl"
    if os.path.exists(car_path):
        try:
            models["car"] = joblib.load(car_path)
            models_loaded["car"] = True
            logger.info("Loaded car model from %s", car_path)
        except Exception as e:
            models["car"] = None
            models_loaded["car"] = False
            logger.exception("Failed to load car model: %s", e)
    else:
        logger.warning("Car model not found at %s", car_path)


tier_1 = [...]
tier_2 = [...]


class HealthUserInput(BaseModel):
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


# CAR input:
class CarUserInput(BaseModel):
    # These are the exact minimal inputs you requested (no extras)
    driver_age: Annotated[int, Field(..., ge=16, le=120)]
    driver_experience: Annotated[int, Field(..., ge=0)]
    previous_accidents: Annotated[int, Field(..., ge=0)]
    annual_mileage_x1000: Annotated[float, Field(..., ge=0.0)]
    car_manufacturing_year: Annotated[int, Field(..., ge=1900, le=2100)]
    car_age: Annotated[int, Field(..., ge=0)]


# Health check endpoint
@app.get("/health")
def health():
    return {
        "models_loaded": {k: v for k, v in models_loaded.items()},
        "sklearn_version": sklearn_version,
    }


# Health predict
@app.post("/health/predict")
def predict_health(data: HealthUserInput):
    if not models_loaded["health"] or models["health"] is None:
        raise HTTPException(status_code=503, detail="Health model not loaded")

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
        model = models["health"]
        pred = model.predict(input_df)[0]
        prob = None
        try:
            prob = float(model.predict_proba(input_df).max())
        except Exception:
            prob = None
    except Exception as e:
        logger.exception("Health prediction error: %s", e)
        raise HTTPException(status_code=500, detail="Prediction failed")

    return JSONResponse(
        content={
            "insurance_type": "health",
            "predicted_category": str(pred),
            "confidence": round(prob, 3) if prob is not None else None,
        }
    )


# Car predict
@app.post("/car/predict")
def predict_car(data: CarUserInput):
    if not models_loaded["car"] or models["car"] is None:
        raise HTTPException(status_code=503, detail="Car model not loaded")

    raw = data.model_dump()
    logger.info("Car raw payload: %s", raw)
    mapped = {
        "Driver Age": raw["driver_age"],
        "Driver Experience": raw["driver_experience"],
        "Previous Accidents": raw["previous_accidents"],
        "Annual Mileage (x1000 km)": raw["annual_mileage_x1000"],
        "Car Manufacturing Year": raw["car_manufacturing_year"],
        "Car Age": raw["car_age"],
    }

    input_df = pd.DataFrame([mapped])
    logger.info("Prepared input_df columns: %s", list(input_df.columns))

    try:
        model = models["car"]
        pred = model.predict(input_df)[0]
        prob = None
        try:
            prob = float(model.predict_proba(input_df).max())
        except Exception:
            prob = None
    except Exception as e:
        logger.exception("Car prediction error: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return JSONResponse(
        content={
            "insurance_type": "car",
            "predicted_category": str(pred),
            "confidence": round(prob, 3) if prob is not None else None,
        }
    )
