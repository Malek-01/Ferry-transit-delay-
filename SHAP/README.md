This folder contains scripts for data analysis and prediction using GradientBoostingRegressor. Each script preprocesses data, trains a model, evaluates its performance, and explains predictions using SHAP values.



Sydney Data Analysis (SHAP-Sydney.py)
This script processes the Data-Sydney.csv file, splitting it into training and testing sets. It trains a gradient boosting model, achieving an R-squared value exceeding 0.7, evaluates the model's performance, and uses SHAP values to explain the predictions. 

Usage
Run with:

```
python SHAP-Sydney.py
```
(To validate these findings, SHAP analysis was also conducted with a Support Vector Regressor (SVR), which produced similarly positive results.)

NYC Data Analysis (SHAP_NYC.py)
This script processes Data-NYC.csv, trains a GradientBoostingRegressor model achieving an R-squared value over 0.75, evaluates its performance, and uses SHAP values to explain predictions.


Usage

```
python SHAP-NYC.py
```
Outputs

Both scripts print evaluation metrics and generate SHAP summary plots saved as PNG files.

Notes

Place Data-Sydney.csv and Data-NYC.csv in the same directory as the scripts or adjust file paths accordingly.
