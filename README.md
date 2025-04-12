Boston Housing Price Prediction 📊🏡
This project focuses on analyzing and modeling the Boston Housing Dataset to predict median house values (medv). The notebook walks through data preprocessing, visualization, outlier handling, and the application of several machine learning regression algorithms.

📁 Dataset
The dataset used is BostonHousing.csv, which includes various features such as crime rate, number of rooms, accessibility to highways, etc., that influence housing prices in Boston suburbs.

🔍 Project Highlights
✅ Data Preprocessing
Loaded and cleaned the dataset

Handled missing values

Performed log transformation to reduce skewness

Removed outliers using IQR method

📊 Data Visualization
Histograms and distribution plots

Box plots before and after log transformation

Correlation heatmap to understand feature relationships

Regression plots for important features vs target variable

🤖 Machine Learning Models
Implemented and evaluated multiple regression models:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

Gradient Boosting Regressor

XGBoost Regressor

Support Vector Regressor (SVR)

Each model's performance is assessed using:

Accuracy (R² Score)

Cross-validation Score

Mean Squared Error

Feature Importance Plots (where applicable)

📈 Goal
The goal is to compare the performance of various regression models and identify the most accurate model for predicting housing prices.

📌 Tools & Libraries
Python

Pandas, NumPy

Seaborn, Matplotlib

Scikit-learn

XGBoost

