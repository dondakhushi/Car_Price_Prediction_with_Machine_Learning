# 🚗 Car Price Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-green.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)

## 📌 Project Overview

Car prices depend on multiple factors such as brand reputation, manufacturing year, fuel type, kilometers driven, transmission type, and ownership history.

This project uses **Machine Learning Regression Techniques** to predict the selling price of a car based on these features. The workflow includes data preprocessing, feature engineering, exploratory data analysis, model training, evaluation, and visualization.

---

## 🎯 Objectives

- Analyze car-related data and identify key price factors.
- Perform data preprocessing and feature engineering.
- Train a regression model for price prediction.
- Evaluate model performance using standard metrics.
- Visualize relationships between features and selling price.
- Understand real-world applications of machine learning in the automobile industry.

---

## 📂 Dataset Information

The dataset contains information about used cars including:

| Feature | Description |
|----------|------------|
| Car_Name | Name of the car |
| Year | Manufacturing year |
| Selling_Price | Selling price of the car (Target Variable) |
| Present_Price | Current showroom price |
| Kms_Driven | Total kilometers driven |
| Fuel_Type | Petrol / Diesel / CNG |
| Seller_Type | Dealer / Individual |
| Transmission | Manual / Automatic |
| Owner | Number of previous owners |

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Jupyter Notebook

---

## 📊 Project Workflow

### 1️⃣ Data Collection
- Load dataset using Pandas.
- Inspect structure and data types.

### 2️⃣ Data Cleaning
- Check missing values.
- Handle inconsistent data.
- Remove unnecessary columns.

### 3️⃣ Feature Engineering
- Convert categorical variables into numerical values.
- Create new features such as car age.

### 4️⃣ Exploratory Data Analysis (EDA)
- Correlation analysis.
- Distribution plots.
- Heatmaps and visualizations.

### 5️⃣ Model Building
- Split data into training and testing sets.
- Train Linear Regression model.

### 6️⃣ Model Evaluation
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

### 7️⃣ Prediction
- Predict car selling prices using unseen data.

---

## 📈 Model Evaluation Metrics

### Mean Absolute Error (MAE)
Measures the average prediction error.

### Mean Squared Error (MSE)
Measures the average squared prediction error.

### R² Score
Indicates how well the model explains the variance in the data.

---

## 📷 Sample Visualizations

### Correlation Heatmap
- Shows relationships between variables.

### Actual vs Predicted Prices
- Compares model predictions with actual values.

---

## 🚀 Installation

Clone the repository:

```bash
https://github.com/dondakhushi/Car_Price_Prediction_with_Machine_Learning.git
