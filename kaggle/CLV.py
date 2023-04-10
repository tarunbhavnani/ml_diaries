# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 12:16:02 2023

@author: ELECTROBOT
https://github.com/hariharan2305/DailyKnowledge/blob/master/Customer%20Lifetime%20Value/Customer%20Lifetime%20Value%20(CLV).ipynb
"""

#clv

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

# Importing the lifetimes package
import lifetimes

# Creating the summary data using summary_data_from_transaction_data function
summary = lifetimes.utils.summary_data_from_transaction_data(data_clv, 'CustomerID', 'InvoiceDate', 'TotalSales' )
summary = summary.reset_index()
summary.head()
#recency of 0 mena sone time buyers

#Create a distribution of frequency to understand the customer frequence level
summary['frequency'].plot(kind='hist', bins=50)
print(summary['frequency'].describe())
print("---------------------------------------")
one_time_buyers = round(sum(summary['frequency'] == 0)/float(len(summary))*(100),2)
print("Percentage of customers purchase the item only once:", one_time_buyers ,"%")


# Fitting the BG/NBD model
bgf = lifetimes.BetaGeoFitter(penalizer_coef=0.0)
bgf.fit(summary['frequency'], summary['recency'], summary['T'])


# Model summary
bgf.summary


# Compute the customer alive probability
summary['probability_alive'] = bgf.conditional_probability_alive(summary['frequency'], summary['recency'], summary['T'])
summary.head(10)


# Visual representation of relationship between recency and frequency
from lifetimes.plotting import plot_probability_alive_matrix

fig = plt.figure(figsize=(12,8))
plot_probability_alive_matrix(bgf)


#Predict future transaction for the next 30 days based on historical dataa
t = 30
summary['pred_num_txn'] = round(bgf.conditional_expected_number_of_purchases_up_to_time(t, summary['frequency'], summary['recency'], summary['T']),2)
summary.sort_values(by='pred_num_txn', ascending=False).head(10).reset_index()

# Let's take CustomerID - 14911, at t=10

# In 372 days, he purchased 131 times. So, in one day he purchases 131/372 = 0.352 times. Hence, for 10 days = 3.52 times.

# Here, our predicted result is 2.98, which is close to the manual probability prediction we did above. The reason 
# for the difference is caused by the various assumptions about the customers, such as the dropout rate, customers 
# lifetime being modeled as exponential distribution, etc.


#the BG/NBD model can only be able to predict the future transactions and churn rate of a customer. In order 
#to add the monetary aspect of the problem, we have to model the monetary value using the Gamma-Gamma Model.

# =============================================================================
# # Some of the key assumptions of Gamma-Gamma model are:
# =============================================================================

# The monetary value of a customer's given transaction varies randomly around their average transaction value.
# Average transaction value varies across customers but do not vary over time for any given customer.
# The distribution of average transaction values across customers is independent of the transaction process.

#lets check if assumptions stand true in our dataset

# Checking the relationship between frequency and monetary_value
return_customers_summary = summary[summary['frequency']>0]
print(return_customers_summary.shape)
return_customers_summary.head()

# Checking the relationship between frequency and monetary_value
return_customers_summary[['frequency', 'monetary_value']].corr()

#The pearson correlation seems very weak. Hence, we can conclude that, the assumption is satisfied and we can fit the model to our data.

# Modeling the monetary value using Gamma-Gamma Model
ggf = lifetimes.GammaGammaFitter(penalizer_coef=0.001)
ggf.fit(return_customers_summary['frequency'],
       return_customers_summary['monetary_value'])

# Summary of the fitted parameters
ggf.summary


#1. model.conditional_expected_average_profit(): This method computes the conditional expectation of the average profit per transaction 
#for a group of one or more customers.

#2. model.customer_lifetime_value(): This method computes the average lifetime value of a group of one or more customers. This method 
#takes in BG/NBD model and the prediction horizon as a parameter to calculate the CLV.

# Calculating the conditional expected average profit for each customer per transaction
summary = summary[summary['monetary_value'] >0]
summary['exp_avg_sales'] = ggf.conditional_expected_average_profit(summary['frequency'],
                                       summary['monetary_value'])
summary.head()


# Checking the expected average value and the actual average value in the data to make sure the values are good
print(f"Expected Average Sales: {summary['exp_avg_sales'].mean()}")
print(f"Actual Average Sales: {summary['monetary_value'].mean()}")


#The values seems to be fine. Now, let's calculate the customer lifetime value directly using the method from the lifetimes package.

# Three main important thing to note here is:

# 1. time: This parameter in customer_lifetime_value() method takes in terms of months i.e., t=1 means one month and so on.

# 2. freq: This parameter is where you will specify the time unit your data is in. If your data is in daily level then "D", 
#monthly "M" and so on.

# 3. discount_rate: This parameter is based on the concept of DCF (discounted cash flow), where you will discount the future 
#monetary value by a discount rate to get the present value of that cash flow. In the documentation, it is given that for 
#monthly it is 0.01 (annually ~12.7%).

# Predicting Customer Lifetime Value for the next 30 days
summary['predicted_clv'] =      ggf.customer_lifetime_value(bgf,
                                                               summary['frequency'],
                                                               summary['recency'],
                                                               summary['T'],
                                                               summary['monetary_value'],
                                                               time=1,     # lifetime in months
                                                               freq='D',   # frequency in which the data is present(T)      
                                                               discount_rate=0.01) # discount rate
summary.head()


#You can also calculate the CLV manually from the predicted number of future transactions (pred_num_txn) and expected 
#average sales per transaction (exp_avg_sales).

summary['manual_predicted_clv'] = summary['pred_num_txn'] * summary['exp_avg_sales']
summary.head()

# CLV in terms of profit (profit margin is 5%)
profit_margin = 0.05
summary['CLV'] = summary['predicted_clv'] * profit_margin
summary.head()


# Distribution of CLV for the business in the next 30 days
summary['CLV'].describe()


