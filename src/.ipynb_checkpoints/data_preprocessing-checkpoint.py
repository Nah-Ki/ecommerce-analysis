import pandas as pd

def load_data():
    customers = pd.read_csv('./data/Customers.csv')
    products = pd.read_csv('./data/Products.csv')
    transactions = pd.read_csv('./data/Transactions.csv')
    return customers, products, transactions

def preprocess_data(customers, products, transactions):
    customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
    transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])
    merged_data = pd.merge(transactions, customers, on='CustomerID')
    merged_data = pd.merge(merged_data, products, on='ProductID')
    return merged_data
