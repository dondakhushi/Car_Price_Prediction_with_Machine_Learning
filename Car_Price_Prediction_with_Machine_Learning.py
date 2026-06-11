#!/workspaces/Car_Price_Prediction_with_Machine_Learning/.venv/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("car data.csv")

print(df.head())

# Explore Dataset
print(df.info())
print(df.describe())
print(df.isnull().sum())

