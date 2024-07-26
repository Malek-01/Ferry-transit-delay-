import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import time

def load_data(filepath):
    """
    Load and preprocess the dataset.

    Parameters:
    filepath (str): The path to the CSV file containing the dataset.

    Returns:
    DataFrame: Loaded dataset.
    """
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    """
    Preprocess the dataset: select features, encode non-numeric columns, and impute missing values.

    Parameters:
    data (DataFrame): The raw dataset.

    Returns:
    DataFrame, Series: Preprocessed features (X) and target (y).
    """
    label_encoder = LabelEncoder()
    
    # Select the relevant features
    X = data.iloc[:, np.r_[1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 17, 19, 20, 23, 24, 25]]
    y = data.iloc[:, 7]
    
    # Encode non-numeric columns
    non_numeric_columns = X.select_dtypes(exclude=['int', 'float']).columns
    X[non_numeric_columns] = X[non_numeric_columns].apply(lambda col: label_encoder.fit_transform(col))
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    return X, y

def remove_outliers(X, y, threshold=3):
    """
    Remove outliers from the dataset based on z-score.

    Parameters:
    X (DataFrame): The features.
    y (Series): The target variable.
    threshold (float): The z-score threshold to identify outliers.

    Returns:
    DataFrame, Series: Features and target with outliers removed.
    """
    z_scores = (X - X.mean()) / X.std()
    outlier_indices = (z_scores > threshold).any(axis=1)
    return X[~outlier_indices], y[~outlier_indices]

def standardize_features(X_train, X_test):
    """
    Standardize the features.

    Parameters:
    X_train (DataFrame): Training features.
    X_test (DataFrame): Testing features.

    Returns:
    array, array: Standardized training and testing features.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def train_and_predict(X_train, y_train, X_test):
    """
    Train the Gradient Boosting Regressor model and make predictions.

    Parameters:
    X_train (array): Standardized training features.
    y_train (Series): Training target.
    X_test (array): Standardized testing features.

    Returns:
    array, float: Predictions and CPU time taken to train and predict.
    """
    start_time = time.process_time()
    reg = GradientBoostingRegressor(learning_rate=0.01, max_depth=3, n_estimators=50)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    end_time = time.process_time()

    cpu_time = end_time - start_time
    return y_pred, cpu_time

def evaluate_model(y_test, y_pred, cpu_time):
    """
    Evaluate the model and print metrics.

    Parameters:
    y_test (Series): True values of the target variable.
    y_pred (array): Predicted values.
    cpu_time (float): CPU time taken to train and predict.
    """
    print("CPU time:", cpu_time, "seconds")
    print('R^2:', r2_score(y_test, y_pred)) 
    print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))  
    print('Mean Squared Error:', mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))

def main():
    """
    Main function to run the data loading, preprocessing, training, and evaluation pipeline.
    """
    data = load_data('Data-Sydney.csv')
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
    
    # Train the model and predict
    y_pred, cpu_time = train_and_predict(X_train_scaled, y_train, X_test_scaled)
    
    # Evaluate the model
    evaluate_model(y_test, y_pred, cpu_time)

if __name__ == "__main__":
    main()
