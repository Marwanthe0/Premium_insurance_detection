# importing required libraries
from fastapi import FastAPI
from pydantic import BaseModel, Field, computed_field
from typing import Literal, Annotated
import pickle
import pandas as pd

# Opening the ML model using pickle
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
# Assigning the app for fastapi
app = FastAPI()


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
        ],
        Field(..., description="Occupation of the user"),
    ]
