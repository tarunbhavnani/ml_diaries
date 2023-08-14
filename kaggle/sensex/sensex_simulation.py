# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 16:35:14 2023

@author: tarun
"""

import pandas as pd

data= pd.read_csv(r"C:\Users\tarun\Desktop\git\ml_diaries\kaggle\sensex\Sensex.csv")

data.loc[:,'Date']=pd.to_datetime(data.Date)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates




fig,ax=plt.subplots(figsize=(12,4))
#ax.plot(data.Date, data.Close)
ax.plot(data['Date'],data['Close'])
ax.set(xlabel='Date',ylabel='Sensex Closing',title='Daily Index Movement')
ax.xaxis.set_major_locator(mdates.YearLocator())



index_returns=data['Close'].pct_change()

import numpy as np
normal_returns= np.random.normal(0, index_returns.dropna().std(), 1294)

import seaborn as sns

fig, ax= plt.subplots()
sns.kdeplot(index_returns.dropna(), ax=ax, label= "index_returns")
sns.kdeplot(normal_returns, ax=ax,label= "normal_returns")
ax.set(title="returns_comparison")
ax.legend()
plt.show()

index_returns.dropna().plot(kind="box")
pd.DataFrame(normal_returns).plot(kind="box")

# =============================================================================
# outliers
# =============================================================================

#index_returns[(index_returns<-.05)]
#index_returns[ (index_returns>.05)]

outliers=data[(index_returns<-0.05) | (index_returns>0.05)]

# =============================================================================
# simulation 
# =============================================================================

#lets simulate the market returrns 1000 times
#each time we will predict the market return for the next 252 days

#lets predict one time

market_sd= index_returns.std()
last_price= data.Close.iloc[-1]
days=252

def sim(last_price,market_sd, days):
    day=0
    sim=[]
    next_price=last_price
    while day<days:
        next_price= next_price+next_price*np.random.normal(0, market_sd)
        sim.append(next_price)
        day+=1
    return sim
        

num_simulations=1000
simulated_data1= pd.DataFrame()
for x in range(num_simulations):
    simulated_data1[x]=sim(last_price,market_sd, days)


#lets plot this
for i in range(100):
    sns.kdeplot(simulated_data1.iloc[:,i])
    #sns.kdeplot(simulated_data.iloc[:,2])
    #sns.kdeplot(simulated_data.iloc[:,3])



fig, ax = plt.subplots()
for i in range(50):
    sns.kdeplot(simulated_data1[i].pct_change().dropna(),ax=ax)




fig=plt.figure()
fig.suptitle('Monte Carlo Simulation Sensex')
plt.plot(simulated_data1)
plt.axhline(y=last_price,color='r',linestyle='-')
plt.xlabel('Day')
plt.ylabel('Index level')







# =============================================================================
# otrher methoid to do the same
# =============================================================================





last_price=data['Close'].iloc[-1]

num_simulations=1000
num_days=252
simulated_data=pd.DataFrame()

for x in range(num_simulations):
    count=0
    daily_volatility=index_returns.std()
    
    price_series=[]
    
    price=last_price*(1+np.random.normal(0,daily_volatility))
    price_series.append(price)
    
    for y in range(num_days):
        if count==251:
            break
        price=price_series[count]*(1+np.random.normal(0,daily_volatility))
        price_series.append(price)
        count+=1
    
    simulated_data[x]=price_series

for i in range(100):
    sns.kdeplot(simulated_data.iloc[:,i])
    #sns.kdeplot(simulated_data.iloc[:,2])
    #sns.kdeplot(simulated_data.iloc[:,3])



fig, ax = plt.subplots()
for i in range(50):
    sns.kdeplot(simulated_data[i].pct_change().dropna(),ax=ax)




fig=plt.figure()
fig.suptitle('Monte Carlo Simulation Sensex')
plt.plot(simulated_data1)
plt.axhline(y=last_price,color='r',linestyle='-')
plt.xlabel('Day')
plt.ylabel('Index level')




# =============================================================================
# potential loss in the value of the position (100,000) based on historical returns and the critical value (alpha) at a 
#confidence level of 0.95. 

#  VaR = position * standard_deviation * z-score

# =============================================================================

import scipy
alpha=scipy.stats.norm.ppf(0.95)
print("Value at 95th percentile:", alpha)




position=100000
var=position*index_returns.dropna().std()*alpha
#var(value at risk)

# =============================================================================
# stress testing
# =============================================================================



daily_volatility= index_returns.std()


# If the market index return volatality increased to 20%
daily_vol_20=((0.20+1)**(1/12))-1
var_20=position*daily_vol_20*alpha

#var simulation onmn simulated data instead of index returns

simulated_volatality=pd.DataFrame(columns=['simulated_volatality','simulated_var'])

for x in range(num_simulations):
    simulated_volatality.loc[x,'simulated_volatality']=simulated_data.iloc[:,x].pct_change().std()
    simulated_volatality.loc[x,'simulated_var']=position*simulated_volatality.loc[x,'simulated_volatality']*alpha

simulated_volatality.simulated_var.plot(kind='box')
sns.kdeplot(simulated_volatality.simulated_var)



# =============================================================================

