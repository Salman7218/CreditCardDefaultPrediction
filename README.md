## Credit Card Default Prediction App

This Flask application implements a web-based interface for predicting the likelihood of credit card default using a pre-trained XGBoost model.

### Features

* **User-friendly interface:** The app provides a simple web form for users to input their credit card details.
* **Pre-trained XGBoost model:** The application utilizes a pre-trained XGBoost model, saved in the `model/xgboost_model.pkl` file.
* **Clear prediction output:** The predicted probability of default is displayed as a percentage.

### Installation

1. **Install Python and necessary libraries:**
   ```bash
   pip install flask
   pip install pandas
   pip install numpy
   pip install pickle
2. **Download the app files:** Download the provided code (this file) and the model/xgboost_model.pkl file.
3. **Run the application:**
   ```bash
   python app.py
# Usage
1.  Access the app: Open your web browser and navigate to http://127.0.0.1:5000/.
2.  Input user details: Fill out the form with the required credit card information.
3.  Submit the form: Click on the "Predict" button.
4.  View the prediction: The predicted probability of default will be displayed on the result page.

#  File Structure
  credit_card_default_prediction/
    ├──Dataset\
    |    └──
    ├──model\
    |    └── xgboost_model.pkl
    ├──templetes\
    |    └── index.html
    |    └──result.html
    ├── app.py
    └──requirements.txt

#  Notes
**Input data:** The app expects input data in the specified format and with 23 columns.
**Pre-trained model:** The pre-trained model should

