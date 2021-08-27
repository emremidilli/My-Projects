import pandas as pd

dfPolicyData = pd.read_csv("PolicyData.csv", delimiter = ";", encoding='latin-1')

dfInvoiceData = pd.read_csv("InvoiceData.csv", delimiter = ";")