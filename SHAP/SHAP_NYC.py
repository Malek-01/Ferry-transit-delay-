# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 17:50:00 2024

@author: Malek
"""
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import shap
import matplotlib.pyplot as plt
import time

# Load the merged data
data = pd.read_csv("Data-NYC.csv")

# Define features and target
X = data.iloc[:, np.r_[0, 1, 3, 4, 5, 6, 7, 10, 18, 21, 22, 23, 24, 25, 26, 27, 28, 29]]
y = data.iloc[:, 8]

# Encode non-numeric columns
label_encoder = LabelEncoder()
non_numeric_columns = X.select_dtypes(exclude=['int', 'float']).columns
X[non_numeric_columns] = X[non_numeric_columns].apply(lambda col: label_encoder.fit_transform(col))

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model on the entire dataset
reg = GradientBoostingRegressor(max_depth=5, n_estimators=200, learning_rate=0.1, random_state=0)
start_time = time.process_time()
reg.fit(X_scaled, y)
end_time = time.process_time()

# Evaluate the model using the entire dataset
y_pred = reg.predict(X_scaled)

# Evaluate the model
cpu_time = end_time - start_time
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

print(f"CPU time: {cpu_time:.2f} seconds")
print('R^2 Score:', r2)
print(f'Mean Absolute Error: {mae:.2f}')
print(f'Mean Squared Error: {mse:.2f}')
print(f'Root Mean Squared Error: {rmse:.2f}')

# Explain predictions using SHAP
explainer = shap.Explainer(reg, X_scaled)
shap_values = explainer(X_scaled, check_additivity=False)  # Disable additivity check

print(f"Shap values length: {len(shap_values)}")
print(f"Sample shap value:\n{shap_values[0]}")

# Visualize SHAP values as a bar plot
shap.summary_plot(shap_values, X, plot_type='bar')
plt.savefig('Shap1_NYC.png')
