# Ferry Transit Delay Prediction

This repository contains code for predicting ferry transit delays using various machine learning models and evaluating their performance.


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

1. Clone the repository:

   ```bash
   git clone <repository_url>
   cd Ferry-transit-delay--main

Install the required libraries (if not already installed):
pip install -r requirements.txt

Run the prediction scripts located in the Prediction directory to predict ferry transit delays using different models. For example, to run the Random Forest prediction:

python Prediction/RF\ prediction.py

Run the SHAP analysis scripts located in the SHAP directory to interpret the model predictions. For example:

python SHAP/SHAP\ Sydney\ .py

Run the cross-validation scripts located in the Validation directory to evaluate model performance. For example:

python Validation/RF\ CV.py

# Ferry Transit Delay Prediction

This repository contains code for predicting ferry transit delays using various machine learning models and evaluating their performance.


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

1. Clone the repository:

   ```bash
   git clone <repository_url>
   cd Ferry-transit-delay--main
Install the required libraries (if not already installed):

bash
pip install -r requirements.txt
Run the prediction scripts located in the Prediction directory to predict ferry transit delays using different models. For example, to run the Random Forest prediction:

bash
python Prediction/RF\ prediction.py
Run the SHAP analysis scripts located in the SHAP directory to interpret the model predictions. For example:

bash
python SHAP/SHAP\ Sydney\ .py
Run the cross-validation scripts located in the Validation directory to evaluate model performance. For example:

bash
python Validation/RF\ CV.py
Contributing
Contributions are welcome! Please open an issue or submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.

