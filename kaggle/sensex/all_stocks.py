# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 11:26:35 2023

@author: tarun
"""

import pandas as pd


data= pd.read_csv(r"C:\Users\tarun\Desktop\git\ml_diaries\kaggle\sensex\all_stocks_5yr.csv")
data.date= pd.to_datetime(data.date)

index_returns=data.close.pct_change()

import seaborn as sns
sns.kdeplot(index_returns)
pd.DataFrame(index_returns).plot(kind="box")

data.Name.value_counts()
data.date.value_counts()

#some test##########################
new=pd.DataFrame(data.groupby('date')["close"].apply(lambda x: sum(x))).reset_index()
new["pct_change"]=new.close.pct_change()
new["pct_change"].fillna(0, inplace=True)

sns.kdeplot(new["pct_change"])

####################################


# =============================================================================
# Let's first use monte carlo simulation for forecasting forr aal
# =============================================================================

data = data[data.Name == 'AAL']

from scipy.stats import norm
import numpy as np
log_returns = np.log(1 + data.close.pct_change())
u = log_returns.mean() #Mean of the logarithmich return
var = log_returns.var() #Variance of the logarithic return
drift = u - (0.5 * var) #drift / trend of the logarithmic return
stdev = log_returns.std() #Standard deviation of the log return


t_intervals = 250 #I just wanted to forecast 250 time points
iterations = 10 #I wanted to have 10 different forecast

daily_returns = np.exp(drift + stdev * norm.ppf(np.random.rand(t_intervals, iterations)))
daily_returns.shape
#daily_returns actually is some kind of a noise. When we multiply this with the t time price, we can obtain t+1 time price

#intial price
S0 = data.close.iloc[-1]

#Let us first create en empty matrix such as daily returns
price_list = np.zeros_like(daily_returns)
price_list[0] = S0
price_list


# With a simple for loop, we are going to forecast the next 250 days
for t in range(1, t_intervals):
    price_list[t] = price_list[t - 1] * daily_returns[t]
price_list = pd.DataFrame(price_list)
price_list['close'] = price_list[0]
price_list.head()


import matplotlib as plt
plt.plot(range(0,250),price_list.loc[:,"close"])
for i in range(10):
    plt.plot(range(0,250),price_list.iloc[:,i])


close = data.close
close = pd.DataFrame(close)
frames = [close, price_list]
monte_carlo_forecast = pd.concat(frames)

monte_carlo = monte_carlo_forecast.iloc[:,:].values
import matplotlib.pyplot as plt
plt.figure(figsize=(17,8))
plt.plot(monte_carlo)
plt.show()


#log rerturns

#arma model
#lstm