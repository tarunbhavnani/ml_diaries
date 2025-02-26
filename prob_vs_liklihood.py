# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:21:13 2023

@author: tarun
"""


# =============================================================================
# create data for coinn flips with heads prob of .6
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli

# Probability of getting heads (success)
p_heads = 0.6

# Simulate coin flips
np.random.seed(41)  # for reproducibility
coin_flips = bernoulli.rvs(p_heads, size=1000)

# Calculate the observed probability (frequency) of heads
observed_probability = np.sum(coin_flips) / len(coin_flips)

# Plot the results
plt.hist(coin_flips, bins=[-0.5, 0.5, 1.5], align='mid', rwidth=0.8, color='skyblue', edgecolor='black')
plt.title('Simulated Coin Flips')
plt.xlabel('Outcome (0: Tails, 1: Heads)')
plt.ylabel('Frequency')
plt.show()

# Display the observed probability
print(f'Observed Probability of Heads: {observed_probability:.2f}')


# =============================================================================
# get liklihood from data of 10 flips and also the adta from above
# =============================================================================
import numpy as np
from scipy.stats import bernoulli
import random
# Observed data (coin flips)
#observed_data = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
observed_data = np.array([random.randint(0,1) for i in range(100)])
#observed_data = coin_flips

# Likelihood function for a Bernoulli distribution
def likelihood(p):
    return np.prod(bernoulli.pmf(observed_data, p))

# Calculate likelihood for different values of p
p_values = np.linspace(0, 1, 100)
likelihood_values = [likelihood(p) for p in p_values]

# Plot the likelihood function
import matplotlib.pyplot as plt

plt.plot(p_values, likelihood_values, label='Likelihood')
plt.title('Likelihood Function for Coin Flip Data')
plt.xlabel('Probability of Heads (p)')
plt.ylabel('Likelihood')
plt.legend()
plt.show()

# =============================================================================
# if we take 1000 coin flips the liklihood comes to zero.
#The likelihood function can become very small, especially for a large number of observations, due to the nature of 
#the product of probabilities. 
#When you multiply many probabilities between 0 and 1, the result can become extremely small. 
#This is often an issue with numerical precision.
# =============================================================================

observed_data = coin_flips

def log_likelihood(p):
    return np.sum(np.log(bernoulli.pmf(observed_data, p)))

# Calculate likelihood for different values of p
p_values = np.linspace(0, 1, 100)
likelihood_values = [likelihood(p) for p in p_values]
likelihood_values = [log_likelihood(p) for p in p_values]

# Plot the likelihood function
import matplotlib.pyplot as plt

plt.plot(p_values, likelihood_values, label='Likelihood')
plt.title('Likelihood Function for Coin Flip Data')
plt.xlabel('Probability of Heads (p)')
plt.ylabel('Likelihood')
plt.legend()
plt.show()

from scipy.optimize import minimize

result = minimize(log_likelihood, x0=0.5, bounds=[(0, 1)])


# =============================================================================

# find the maximum likelihood for 10 and all , notice how big data gives .5 and not .6

#This MLE for p is the point around which the likelihood function tends to concentrate, indicating the 
#most likely value 
#for the probability of getting heads based on the observed data.


# =============================================================================

import numpy as np
from scipy.stats import bernoulli
from scipy.optimize import minimize

# Observed data (coin flips)
observed_data = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
#observed_data = list(coin_flips)

# Likelihood function for a Bernoulli distribution
def negative_likelihood(p):
    return -np.prod(bernoulli.pmf(observed_data, p))

# Use optimization to find the value of p that maximizes the likelihood
result = minimize(negative_likelihood, x0=0.5, bounds=[(0, 1)])

# The MLE for p is the value that maximizes the likelihood
mle_p = result.x[0]

print(f'Maximum Likelihood Estimate (MLE) for p: {mle_p:.4f}')






# =============================================================================
# take random samples of 10 from all 1000 data, find maximum liklihhod and plot to get the real thing
# =============================================================================
np.random.seed(41)  # for reproducibility
coin_flips = bernoulli.rvs(p_heads, size=1000)

p_values = np.linspace(0, 1, 100)

def negative_likelihood(p):
    return -np.prod(bernoulli.pmf(observed_data, p))


mles=[]
for i in range(1000):
    observed_data = np.random.choice(coin_flips, size=10, replace=False)
    
    # Use optimization to find the value of p that maximizes the likelihood
    result = minimize(negative_likelihood, x0=0.5, bounds=[(0, 1)])

    # The MLE for p is the value that maximizes the likelihood
    mle_p = result.x[0]
    mles.append(mle_p)

import seaborn as sns
sns.kdeplot(mles)
    
# =============================================================================
# mle 
# =============================================================================


import numpy as np
from scipy.stats import norm

# Generate some example data
np.random.seed(421)
data = np.random.normal(loc=2, scale=1, size=100)

# Define the likelihood function for a normal distribution
def likelihood(params):
    mean, std_dev = params
    return -np.sum(norm.logpdf(data, loc=mean, scale=std_dev))

# Initial guess for the parameters
initial_guess = [0, 1]

# Use an optimization algorithm to find the parameter values that maximize the likelihood
from scipy.optimize import minimize

result = minimize(likelihood, initial_guess, method='L-BFGS-B')

# Extract the optimized parameters
optimized_params = result.x

print("Optimized Parameters (Mean, Standard Deviation):", optimized_params)
