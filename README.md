# Ferry Transit Delay Prediction
This repository contains code for predicting ferry transit delays using various machine learning models and evaluating their performance. The data used in this project is available in a shared Google Drive folder. You can access it [here](https://drive.google.com/drive/folders/1Qocy3x_pePx0oHZjX_QpbubIB_8KDNZX?usp=drive_link).


## Requirements

- Python 3.x
- Dependencies: The required Python libraries are specified in the requirements.txt file

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

Run the cross-validation scripts located in the Validation directory to optimize hyperparameters for each model and evaluate their performance. This step ensures that you select the best model configuration for accurate ferry transit delay predictions. For example, to run the Random Forest:
```
python Validation/RF-CV.py
```

Execute the prediction scripts located in the Prediction directory to forecast ferry transit delays using various pre-configured machine learning techniques. 
```
python Prediction/RF-prediction.py
```

Run the SHAP analysis scripts located in the SHAP directory to interpret model predictions and identify the most important features affecting those predictions. SHAP (SHapley Additive exPlanations) values offer a comprehensive view of feature contributions. For example:
```
python SHAP/SHAP-Sydney.py
```


## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

For any questions or inquiries, please contact the main contributor:

- Malek Sarhani: malek.sarhani.aui@gmail.com or m.sarhani@aui.ma
