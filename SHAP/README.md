This folder contains scripts for data analysis and prediction using RandomForestRegressor. Each script preprocesses data, trains a model, evaluates its performance, and explains predictions using SHAP values.



Sydney Data Analysis (SHAP-Sydney.py)
This script processes data from Data-Sydney.csv, splits it into training and testing sets, trains a RandomForestRegressor model on the entire model (R-square more than 0.9), evaluates its performance, and explains predictions using SHAP.

Usage
Run with:

```
python SHAP-Sydney.py
```

NYC Data Analysis (SHAP_NYC.py)
This script processes data from Data-NYC.csv, trains a RandomForestRegressor model on the entire dataset, evaluates its performance (R-square more than 0.9), and explains predictions using SHAP.

Usage

```
python SHAP-NYC.py
```
Outputs
Both scripts print evaluation metrics and generate SHAP summary plots saved as PNG files.

Notes
Place Data-Sydney.csv and Data-NYC.csv in the same directory as the scripts or adjust file paths accordingly.
