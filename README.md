# Customer Churn Detection Model

![image6](https://user-images.githubusercontent.com/81787449/132009565-9b5f6f66-adfd-4b55-b1b7-754ca82da931.png)

## Project Overview

This project aims to predict customer churn using machine learning models. Customer churn refers to the loss of customers in a business. By predicting which customers are likely to churn, businesses can take proactive measures to retain them. The project involves data preprocessing, model training, evaluation, and deployment of the final model using Flask.

## Dataset Summary

The dataset used for this project contains the following columns:

- **CreditScore**: The credit score of the customer.
- **Geography**: The country from which the customer belongs.
- **Gender**: The gender of the customer (Male/Female).
- **Age**: The age of the customer.
- **Tenure**: The number of years the customer has been with the bank.
- **Balance**: The balance left in the customer’s account.
- **NumOfProducts**: The number of products the customer is using.
- **HasCrCard**: Whether the customer has a credit card (1 = Yes, 0 = No).
- **IsActiveMember**: Whether the customer is an active member (1 = Yes, 0 = No).
- **EstimatedSalary**: The estimated salary of the customer.

## Model Training and Evaluation

Two machine learning models were trained and evaluated on this dataset:

1. **Random Forest Classifier**
2. **XGBoost Classifier**

After evaluating the models based on various metrics, including the F1-score, the **XGBoost Classifier** was chosen as the final model due to its superior performance.

## Deployment

The final model was deployed using Flask, allowing for real-time predictions on new data. The Flask application provides an interface to input customer details and returns the likelihood of churn, enabling businesses to make informed decisions.

## How to Run the Project

1. **Clone the repository**:
   ```bash
   git clone <repository-link>
2. **Navigate to the project directory:**
    ```bash
    cd Customer_Churn_Model
3. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
4. **Run the Flask application:**
    ```bash
    python router_flask.py
5. **Access the application:** Open your browser and navigate to `http://127.0.0.1:5000/` to use the customer churn prediction model.

## Conclusion
This project demonstrates an end-to-end pipeline for customer churn detection, from data preprocessing to model deployment. The XGBoost model was selected as the final model due to its high F1-score, and the model was successfully deployed using Flask for real-time predictions.
