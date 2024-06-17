from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model/xgboost_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Gather input features from the form
            input_data = {
                'LIMIT_BAL': float(request.form['limit_bal']),
                'SEX': int(request.form['sex']),
                'EDUCATION': int(request.form['education']),
                'MARRIAGE': int(request.form['marriage']),
                'AGE': int(request.form['age']),
                'PAY_0': int(request.form['pay_0']),
                'PAY_2': int(request.form['pay_2']),
                'PAY_3': int(request.form['pay_3']),
                'PAY_4': int(request.form['pay_4']),
                'PAY_5': int(request.form['pay_5']),
                'PAY_6': int(request.form['pay_6']),
                'BILL_AMT1': float(request.form['bill_amt1']),
                'BILL_AMT2': float(request.form['bill_amt2']),
                'BILL_AMT3': float(request.form['bill_amt3']),
                'BILL_AMT4': float(request.form['bill_amt4']),
                'BILL_AMT5': float(request.form['bill_amt5']),
                'BILL_AMT6': float(request.form['bill_amt6']),
                'PAY_AMT1': float(request.form['pay_amt1']),
                'PAY_AMT2': float(request.form['pay_amt2']),
                'PAY_AMT3': float(request.form['pay_amt3']),
                'PAY_AMT4': float(request.form['pay_amt4']),
                'PAY_AMT5': float(request.form['pay_amt5']),
                'PAY_AMT6': float(request.form['pay_amt6'])
            }
            
            # Convert input data into a DataFrame
            input_df = pd.DataFrame([input_data])

            if input_df.shape[1] != 23:
                return "Error: Input data shape mismatch. Expected 24 columns."

            # Make prediction using the model
            prediction = model.predict_proba(input_df)[:, 1]  # Probability of default

            # Format prediction as a percentage
            prediction_percentage = prediction[0] * 100

            return render_template('result.html', prediction=f"{prediction_percentage:.2f}%")
        except Exception as e:
            return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
