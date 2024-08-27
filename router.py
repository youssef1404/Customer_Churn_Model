import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template



## Intialize
app = Flask(__name__)


# Home
@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')


# Predict
@app.route('/predict')
def predict():
    return render_template('predict.html')

# Predict
@app.route('/about')
def about():
    return render_template('about.html')


# Terminal
if __name__ == '__main__':
    app.run(debug=True)