import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request
from utils import predict_new2
from pydantic import BaseModel, Field



# Columns in order as user input
columns = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
        'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'withdrawing']


# Define desired valid dtypes
dtypes = {
    'CreditScore': float,
    'Geography': str,
    'Gender': str,
    'Age': int,
    'Tenure': int,
    'Balance': float,
    'NumOfProducts': int,
    'HasCrCard': int,
    'IsActiveMember': int,
    'EstimatedSalary': float,
    'withdrawing': float
}

## Intialize
app = Flask(__name__)


# Home
@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')


# Predict
@app.route('/predict', methods=['GET', 'POST'])
def predict(): # while prediction
    if request.method == 'POST':
        credit = float(request.form['credit'])
        Geography = request.form['geoo']
        gender = request.form['gender']
        age = int(request.form['age'])
        tenure = int(request.form['tenure'])
        balance = float(request.form['Balance'])
        NumOfProducts = int(request.form['nums'])
        HasCrCard = int(request.form['HasCrCard'])
        IsActiveMember = int(request.form['IsActiveMember'])
        EstimatedSalary = float(request.form['salary'])

        withdrawing = abs(balance - EstimatedSalary)

        # Concatenate all features form Pydantic
        input_data = np.array([credit, Geography, gender, age,
                            tenure, balance, NumOfProducts, HasCrCard,
                            IsActiveMember, EstimatedSalary, withdrawing])
        
        # Adjust the column names and dtypes
        input_data = pd.DataFrame([input_data], columns=columns)
        X_new = input_data.astype(dtypes)
        y_pred = predict_new2(X_new)

        return render_template('predict.html', y_pred=y_pred)
    else:
        return render_template('predict.html')

# Predict
@app.route('/about')
def about():
    return render_template('about.html')


# Terminal
if __name__ == '__main__':
    app.run(debug=True)