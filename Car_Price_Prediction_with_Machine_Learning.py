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

# Data Preprocessing
# Convert Categorical Data into Numbers
le = LabelEncoder()

df['Fuel_Type'] = le.fit_transform(df['Fuel_Type'])
df['Selling_type'] = le.fit_transform(df['Selling_type'])
df['Transmission'] = le.fit_transform(df['Transmission'])

# Create Car Age Feature
df['Current_Year'] = 2025
df['Car_Age'] = df['Current_Year'] - df['Year']

df.drop(['Year', 'Car_Name', 'Current_Year'], axis=1, inplace=True)

# Visualization
# Correlation Heatmap       
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show() 

 # Define Features and Target                  
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price'] 

# Split Dataset                    
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)                         

# Train Regression Model                        
model = LinearRegression()
model.fit(X_train, y_train)  

# Make Predictions                                         
y_pred = model.predict(X_test)

print(y_pred[:5])

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE :", mae)
print("MSE :", mse)
print("R2 Score :", r2)

# Actual vs Predicted Graph
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.show()