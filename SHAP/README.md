This folder contains scripts for data analysis and prediction using RandomForestRegressor. Each script preprocesses data, trains a model, evaluates its performance, and explains predictions using SHAP values.



Sydney Data Analysis (SHAP-Sydney.py)
This script processes Data-Sydney.csv, splitting it into training and testing sets. It trains a RandomForestRegressor model, achieving an R-squared value over 0.9, evaluates performance, and uses SHAP values to explain predictions.

Usage
Run with:

```
python SHAP-Sydney.py
```

NYC Data Analysis (SHAP_NYC.py)
This script processes Data-NYC.csv, trains a RandomForestRegressor model achieving an R-squared value over 0.9, evaluates its performance, and uses SHAP values to explain predictions.


Usage

```
python SHAP-NYC.py
```
Outputs
Both scripts print evaluation metrics and generate SHAP summary plots saved as PNG files.

Notes
Place Data-Sydney.csv and Data-NYC.csv in the same directory as the scripts or adjust file paths accordingly.
