import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
label_encoder = LabelEncoder()
data = pd.read_csv('data3.csv')
data = data.iloc[1:100000,:]
X = data.iloc[:, np.r_[ 1,2, 3,4, 5, 6,  8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,  23, 24, 25]]
y = data.iloc[:, 7]
print(X.columns)
print("y")
print(y)


# Encode all non-numeric columns
non_numeric_columns = X.select_dtypes(exclude=['int', 'float']).columns
X[non_numeric_columns] = X[non_numeric_columns].apply(lambda col: label_encoder.fit_transform(col))

train_pct_index = int(0.8 * len(data.iloc[:, 2]))
X_train, X_test = X.iloc[:train_pct_index], X.iloc[train_pct_index:]
y_train, y_test = y.iloc[:train_pct_index], y.iloc[train_pct_index:]
# Identify and remove outliers from the training set
z_scores_train = (X_train - X_train.mean()) / X_train.std()
outlier_threshold = 3  # Adjust this threshold based on your data
outlier_indices_train = (z_scores_train > outlier_threshold).any(axis=1)
X_train = X_train[~outlier_indices_train]
y_train = y_train[~outlier_indices_train]

# Identify and remove outliers from the test set
z_scores_test = (X_test - X_test.mean()) / X_test.std()
outlier_indices_test = (z_scores_test > outlier_threshold).any(axis=1)
X_test = X_test[~outlier_indices_test]
y_test = y_test[~outlier_indices_test]

# Standardizing features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
rf_model = RandomForestRegressor()
    # Specify hyperparameters to tune
param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
    }
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)
# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print('R:', r2_score(y_test, y_pred)) 
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))