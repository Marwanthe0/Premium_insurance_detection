# importing required libraries
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, computed_field
from typing import Literal, Annotated
import pickle
import pandas as pd

# Opening the ML model using pickle
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
# Assigning the fastapi object in the app
app = FastAPI()

# List of cities by their tiers
tier_1 = [
    "Mumbai",
    "Delhi",
    "Bangalore",
    "Chennai",
    "Kolkata",
    "Hyderabad",
    "Pune",
    "Dhaka",
    "Sylhet",
    "Chattogram",
]
tier_2 = [
    "Khulna",
    "Rajshahi",
    "Barishal",
    "Rangpur",
    "Mymensingh",
    "Comilla",
    "Jaipur",
    "Chandigarh",
    "Indore",
    "Lucknow",
    "Patna",
    "Ranchi",
    "Visakhapatnam",
    "Coimbatore",
    "Bhopal",
    "Nagpur",
    "Vadodara",
    "Surat",
    "Rajkot",
    "Jodhpur",
    "Raipur",
    "Amritsar",
    "Varanasi",
    "Agra",
    "Dehradun",
    "Mysore",
    "Jabalpur",
    "Guwahati",
    "Thiruvananthapuram",
    "Ludhiana",
    "Nashik",
    "Allahabad",
    "Udaipur",
    "Aurangabad",
    "Hubli",
    "Belgaum",
    "Salem",
    "Vijayawada",
    "Tiruchirappalli",
    "Bhavnagar",
    "Gwalior",
    "Dhanbad",
    "Bareilly",
    "Aligarh",
    "Gaya",
    "Kozhikode",
    "Warangal",
    "Kolhapur",
    "Bilaspur",
    "Jalandhar",
    "Noida",
    "Guntur",
    "Asansol",
    "Siliguri",
]


# Data Validation using pydantic_class
class UserInput(BaseModel):
    age: Annotated[
        int, Field(..., gt=0, lt=120, description="This is the Age of the user")
    ]
    weight: Annotated[float, Field(..., gt=0, description="Weight of the user")]
    height: Annotated[
        float, Field(..., gt=0, description="This is the Height of the user")
    ]
    income_lpa: Annotated[
        float, Field(..., gt=0, description="Yearly income of the user")
    ]
    smoker: Annotated[bool, Field(..., description="Is the user a smoker?")]
    city: Annotated[str, Field(..., description="City of the user")]
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
        Field(..., description="Occupation of the user"),
    ]

    # generating features from the user input
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


@app.post("/predict")
def predict_premium(data: UserInput):
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
    prediction = model.predict(input_df)[0]
    return JSONResponse(
        status_code=200, content={"response": {"predicted_category": prediction}}
    )
