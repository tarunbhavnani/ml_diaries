# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 18:34:55 2023

@author: ELECTROBOT
"""
import pandas as pd
import numpy as np
from datetime import datetime 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

data= pd.read_csv(r"C:\Users\ELECTROBOT\Desktop\MIT-Optmization\Churn\data.csv", encoding="ISO-8859-1")
list(data)
hd= data.head()

# Feature selection
features = ['CustomerID', 'InvoiceNo', 'InvoiceDate', 'Quantity', 'UnitPrice']
data_clv = data[features]
data_clv['TotalSales'] = data_clv['Quantity'].multiply(data_clv['UnitPrice'])
print(data_clv.shape)
data_clv.head()

data_clv['InvoiceDate']=[datetime.strptime(i.split()[0], "%m/%d/%Y") for i in data_clv['InvoiceDate']]


len(set(data_clv.InvoiceNo))==len(data_clv)

#there are invoices with different items thus multiple observations for same invoice, we club them with mean

data_clv=data_clv.groupby(['CustomerID', 'InvoiceNo', 'InvoiceDate']).agg({'Quantity':lambda x: sum(x),
                                                                  'UnitPrice':lambda x: np.mean(x),
                                                                  'TotalSales':lambda x: sum(x)}).reset_index()



data_clv.describe()


#remove negatives(refunds/returns)
data_clv = data_clv[data_clv['TotalSales'] > 0]
data_clv.describe()

#lets check negative!

data_clv.isnull().sum()

#drop cid missing

data_clv=data_clv[~data_clv['CustomerID'].isnull()]


# Printing the details of the dataset
maxdate = data_clv['InvoiceDate'].dt.date.max()
mindate = data_clv['InvoiceDate'].dt.date.min()
unique_cust = data_clv['CustomerID'].nunique()
tot_quantity = data_clv['Quantity'].sum()
tot_sales = data_clv['TotalSales'].sum()

print(f"The Time range of transactions is: {mindate} to {maxdate}")
print(f"Total number of unique customers: {unique_cust}")
print(f"Total Quantity Sold: {tot_quantity}")
print(f"Total Sales for the period: {tot_sales}")


# =============================================================================
# Aggregate Model
#CLV = ((Average Sales X Purchase Frequency) / Churn) X Profit Margin
# =============================================================================





# Transforming the data to customer level for the analysis


# data_clv.groupby("CustomerID")["InvoiceDate"].apply(lambda x:(x.max()-x.min()).days )
# data_clv.groupby("CustomerID")["InvoiceNo"].apply(lambda x:len(x) )
# data_clv.groupby("CustomerID")["TotalSales"].apply(lambda x:sum(x) )


customer = data_clv.groupby('CustomerID').agg({'InvoiceDate':lambda x: (x.max() - x.min()).days, 
                                                   'InvoiceNo': lambda x: len(x),
                                                  'TotalSales': lambda x: sum(x)})

customer.columns = ['Age', 'Frequency', 'TotalSales']
customer.head()

# Calculating the necessary variables for CLV calculation
Average_sales = round(np.mean(customer['TotalSales']),2)
print(f"Average sales: ${Average_sales}")

Purchase_freq = round(np.mean(customer['Frequency']), 2)
print(f"Purchase Frequency: {Purchase_freq}")

Retention_rate = customer[customer['Frequency']>1].shape[0]/customer.shape[0]
churn = round(1 - Retention_rate, 2)
print(f"Churn: {churn}%")

# Calculating the CLV
Profit_margin = 0.05

CLV = round(((Average_sales * Purchase_freq/churn)) * Profit_margin, 2)
print(f"The Customer Lifetime Value (CLV) for each customer is: ${CLV}")



# =============================================================================
# Cohort Model
# =============================================================================


#Transforming the data to customer level for the analysis
customer = data_clv.groupby('CustomerID').agg({'InvoiceDate':lambda x: x.min().month, 
                                                   'InvoiceNo': lambda x: len(x),
                                                  'TotalSales': lambda x: np.sum(x)})

customer.columns = ['Start_Month', 'Frequency', 'TotalSales']
customer.head()
customer["test"]= customer['TotalSales']/customer['Frequency']

#set([i.month for i in data_clv["InvoiceDate"]])
#data_clv.groupby('CustomerID')['InvoiceDate'].apply(lambda x: sorted(list(set([i.month for i in x]))))


# Calculating CLV for each cohort
months = ['Jan', 'Feb', 'March', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
Monthly_CLV = []

for i in range(1, 13):
    customer_m = customer[customer['Start_Month']==i]
    
    Average_sales = round(np.mean(customer_m['TotalSales']),2)
    
    Purchase_freq = round(np.mean(customer_m['Frequency']), 2)
    
    Retention_rate = customer_m[customer_m['Frequency']>1].shape[0]/customer_m.shape[0]
    churn = round(1 - Retention_rate, 2)
    
    CLV = round(((Average_sales * Purchase_freq/churn)) * Profit_margin, 2)
    
    Monthly_CLV.append(CLV)


# =============================================================================
# BG/NBD Model (with Gamma-Gamma extension)
#BG/NBD stands for Beta Geometric/Negative Binomial Distribution.
#alternative to Pareto/NBD
#This is one of the most commonly used probabilistic model for predicting the CLV
# The BG/NBD model has few assumptions:

# When a user is active, number of transactions in a time t is described by Poisson distribution with rate lambda.

# Heterogeneity in transaction across users (difference in purchasing behavior across users) has Gamma distribution with
#  shape parameter r and scale parameter a.

# Users may become inactive after any transaction with probability p and their dropout point is distributed between 
# purchases with Geometric distribution.

# Heterogeneity in dropout probability has Beta distribution with the two shape parameters alpha and beta.

# Transaction rate and dropout probability vary independently across users.
# =============================================================================



#https://lifetimes.readthedocs.io/en/latest/Quickstart.html
#create data for  BG/NBD model

last_day= data_clv.InvoiceDate.max()#'InvoiceDate':lambda x: (last_day-x.max()).days,

#we have divided by 30 below, thus making all this month wise, remove 30 and it becomes day, make it 90 and it becomes quarter!

customer = data_clv.groupby('CustomerID').agg({'InvoiceDate':lambda x: round((x.max()-x.min()).days/30), 
                                                   'InvoiceNo': lambda x: len(x)-1,
                                                  'TotalSales': lambda x: np.sum(x)/len(x)})
customer.columns = ['recency', 'frequency', 'monetary_value']
customer["T"]=data_clv.groupby('CustomerID')['InvoiceDate'].apply(lambda x: round((last_day-x.min()).days/30))


from lifetimes import BetaGeoFitter

# similar API to scikit-learn and lifelines.
bgf = BetaGeoFitter(penalizer_coef=0.0)
bgf.fit(customer['frequency'], customer['recency'], customer['T'])
print(bgf)

bgf.summary

from lifetimes.plotting import plot_frequency_recency_matrix

plot_frequency_recency_matrix(bgf)


from lifetimes.plotting import plot_probability_alive_matrix

plot_probability_alive_matrix(bgf)

#value of t below determines how many time periods are you predicting for!
t = 12
customer['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(t, customer['frequency'], customer['recency'], customer['T'])
customer.sort_values(by='predicted_purchases').tail(5)


from lifetimes.plotting import plot_period_transactions
plot_period_transactions(bgf)





# =============================================================================
# do manual simulations , remove last one month and predict for next one month
# =============================================================================









