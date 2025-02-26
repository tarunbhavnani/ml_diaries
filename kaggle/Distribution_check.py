# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 19:04:50 2024

@author: tarun
"""

#is it a log normal distribution?


#lets create a log normal distribution first, exponential and normal as well

import numpy as np

data_ln= np.random.lognormal(mean=0, sigma=1, size=1000)
data_normal= np.random.normal(0,1,1000)
data_ex=np.random.exponential(1,1000)




import seaborn as sns
sns.kdeplot(data_ln)
sns.kdeplot(data_normal)
sns.kdeplot(data_ex)


import matplotlib.pyplot as plt
from scipy.stats import probplot


#check for normal
probplot(data_ex, dist="norm", sparams=(0,1), plot=plt)
probplot(data_ln, dist="norm", sparams=(0,1), plot=plt)
probplot(data_normal, dist="norm", sparams=(0,1), plot=plt)

#check for exponential
probplot(data_ex, dist="expon", sparams=(0,1), plot=plt)
probplot(data_ln, dist="expon", sparams=(0,1), plot=plt)
probplot(data_normal, dist="expon", sparams=(0,1), plot=plt)

#check for log normal
probplot(data_ex, dist="lognorm", sparams=(1,0), plot=plt)
probplot(data_ln, dist="lognorm", sparams=(1,0), plot=plt)
probplot(data_normal, dist="lognorm", sparams=(1,0), plot=plt)


#check the graphs, the distribution following the distributioon will hug the straight line!!


