import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
import shap
import time

def load_and_preprocess_data(filepath):
    """Load and preprocess the data."""
    label_encoder = LabelEncoder()
    data = pd.read_csv(filepath)
    data = data.rename(columns={'current_stop_sequence': 'stop_sequence'})

    # Select features and target variable
    X = data.iloc[:, np.r_[1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 17, 19, 20, 23, 24, 25]]
    y = data.iloc[:, 7]

    # Fill missing values in specific columns
    X.iloc[:, 4].fillna(0, inplace=True)
    X.iloc[:, 5].fillna(0, inplace=True)

    # Encode non-numeric columns
    non_numeric_columns = X.select_dtypes(exclude=['int', 'float']).columns
    X[non_numeric_columns] = X[non_numeric_columns].apply(lambda col: label_encoder.fit_transform(col))

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    return X, y

def split_and_scale_data(X, y, train_size=0.8):
    """Split the data into training and testing sets and scale the features."""
    train_pct_index = int(train_size * len(X))
    X_train, X_test = X.iloc[:train_pct_index], X.iloc[train_pct_index:]
    y_train, y_test = y.iloc[:train_pct_index], y.iloc[train_pct_index:]

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """Train the model, make predictions, and evaluate the performance."""
    start_time = time.process_time()

    reg = GradientBoostingRegressor(max_depth=5, n_estimators=200, learning_rate=0.1, random_state=0)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    end_time = time.process_time()
    cpu_time = end_time - start_time

    # Model evaluation
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"CPU time: {cpu_time:.2f} seconds")
    print(f'R2 Score: {r2:.2f}')
    print(f'Mean Absolute Error: {mae:.2f}')
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'Root Mean Squared Error: {rmse:.2f}')

    return reg, X_test

def explain_model(reg, X_test):
    """Generate SHAP values and plots."""
    explainer = shap.Explainer(reg)
    shap_values = explainer(X_test)

    print(f"Shap values length: {len(shap_values)}")
    print(f"Sample shap value:\n{shap_values[0]}")

    shap.summary_plot(shap_values)
    shap.summary_plot(shap_values, plot_type='bar')

def main():
    X, y = load_and_preprocess_data('Data-Sydney.csv')
    X_train, X_test, y_train, y_test = split_and_scale_data(X, y)
    reg, X_test_scaled = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    explain_model(reg, X_test_scaled)

if __name__ == "__main__":
    main()
