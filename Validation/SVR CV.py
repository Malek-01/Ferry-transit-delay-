import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def load_data(filepath):
    """Load and preprocess the dataset."""
    data = pd.read_csv(filepath)
    data = data.iloc[1:100000, :]  
    return data

def preprocess_data(data):
    """Preprocess the dataset: encode non-numeric columns."""
    label_encoder = LabelEncoder()
    
    X = data.iloc[:, np.r_[1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25]]
    y = data.iloc[:, 7]
    
    # Encode non-numeric columns
    non_numeric_columns = X.select_dtypes(exclude=['int', 'float']).columns
    X[non_numeric_columns] = X[non_numeric_columns].apply(lambda col: label_encoder.fit_transform(col))
    
    return X, y

def remove_outliers(X, y, threshold=3):
    """Remove outliers from the dataset based on z-score."""
    z_scores = (X - X.mean()) / X.std()
    outlier_indices = (z_scores > threshold).any(axis=1)
    return X[~outlier_indices], y[~outlier_indices]

def standardize_features(X_train, X_test):
    """Standardize the features."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def train_model(X_train, y_train):
    """Train a Support Vector Regressor (SVR) model with GridSearchCV."""
    svr_model = SVR()
    param_grid = {
        'C': [0.1, 1, 100],
        'epsilon': [0.1, 0.2, 0.5],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
    }
    grid_search = GridSearchCV(estimator=svr_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(X_train, y_train)
    return grid_search

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print metrics."""
    y_pred = model.predict(X_test)
    print('R^2:', r2_score(y_test, y_pred))
    print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))

def main():
    data = load_data('data3.csv')
    X, y = preprocess_data(data)
    
    # Split the data
    train_pct_index = int(0.8 * len(X))
    X_train, X_test = X.iloc[:train_pct_index], X.iloc[train_pct_index:]
    y_train, y_test = y.iloc[:train_pct_index], y.iloc[train_pct_index:]
    
    # Remove outliers
    X_train, y_train = remove_outliers(X_train, y_train)
    X_test, y_test = remove_outliers(X_test, y_test)
    
    # Standardize features
    X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)
    
    # Train the model
    grid_search = train_model(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    
    print("Best Hyperparameters:", grid_search.best_params_)
    
    # Evaluate the model
    evaluate_model(best_model, X_test_scaled, y_test)

if __name__ == "__main__":
    main()
