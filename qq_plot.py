# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 11:37:44 2024

@author: tarun

create a log normal distribution , it looks like a normal. use qq plot to acertain its not a normal
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import probplot, lognorm, norm

# Generate example data from a log-normal distribution
np.random.seed(42)
data_lognormal = np.random.lognormal(mean=0, sigma=1, size=1000)

import seaborn as sns
sns.kdeplot(data_lognormal)

# Create a QQ plot comparing against a normal distribution
probplot(data_lognormal, dist='norm', plot=plt)
plt.title('QQ Plot: Log-Normal vs. Normal Distribution')
plt.xlabel('Theoretical Quantiles (Normal)')
plt.ylabel('Observed Quantiles (Log-Normal)')
plt.show()

# Create a QQ plot comparing against a lognormal distribution
probplot(data_lognormal, dist='lognorm',sparams=(1,), plot=plt)
plt.title('QQ Plot: Log-Normal vs. Normal Distribution')
plt.xlabel('Theoretical Quantiles (Normal)')
plt.ylabel('Observed Quantiles (Log-Normal)')
plt.show()

