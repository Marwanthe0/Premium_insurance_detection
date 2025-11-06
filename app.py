#importing required libraries
from fastapi import FastAPI
from pydantic import BaseModel, Field, computed_field
from typing import Literal
import pickle
import pandas as pd
#Opening the ML model using pickle
with open("model.pkl","rb") as f:
    model = pickle.load(f)
#Assigning the app for fastapi
app = FastAPI()