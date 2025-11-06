#importing neseccary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score,classification_report
import numpy as np

#loading the dataset
df = pd.read_csv('insurance.csv')
# df.sample(5)
