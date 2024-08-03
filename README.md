# Ferry Transit Delay Prediction
This repository contains code for predicting ferry transit delays using various machine learning models and evaluating their performance. The data for this project is available here.
https://drive.google.com/drive/folders/1Qocy3x_pePx0oHZjX_QpbubIB_8KDNZX?usp=drive_link


## Requirements

- Python 3.x
- Required Python libraries are specified in each script. Typically, you will need:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `tensorflow` (for DNN)
  - `xgboost` (for Gradient Boosting)
  - `shap` (for SHAP analysis)

## Usage

Usage
Clone the repository:

```bash
git clone https://github.com/Malek-01/Ferry-transit-delay-
cd Ferry-transit-delay--main
```


Install the required libraries (if not already installed):
```
pip install -r requirements.txt
```
Run the prediction scripts located in the Prediction directory to predict ferry transit delays using different models. For example, to run the Random Forest prediction:
```
python Prediction/RF_prediction.py
```

Run the SHAP analysis scripts located in the SHAP directory to interpret the model predictions. For example:
```
python SHAP/SHAP_Sydney.py
```

Run the cross-validation scripts located in the Validation directory to evaluate model performance. For example:
```
python Validation/RF_CV.py
```

Contributing

Contributions are welcome! Please open an issue or submit a pull request.

License

This project is licensed under the MIT License. See the LICENSE file for details.

