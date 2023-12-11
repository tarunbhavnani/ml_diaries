# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 11:05:59 2023

@author: tarun
"""

# =============================================================================
# basic scenario analysis 
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

# Parameters
initial_exposure = 100  # Initial exposure to the counterparty
num_scenarios = 1000   # Number of scenarios
mean_growth = 0.02     # Mean annual revenue growth rate
std_dev_growth = 0.05  # Standard deviation of annual revenue growth rate

# Generate random scenarios for annual revenue growth
np.random.seed(41)  # for reproducibility
annual_growth_rates = np.random.normal(loc=mean_growth, scale=std_dev_growth, size=num_scenarios)

# Calculate cumulative revenue for each scenario
cumulative_revenue = np.cumprod(1 + annual_growth_rates)

# Calculate credit exposure for each scenario
credit_exposure = initial_exposure * cumulative_revenue

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(credit_exposure, label='Credit Exposure')
plt.axhline(y=initial_exposure, color='r', linestyle='--', label='Initial Exposure')
plt.xlabel('Scenario')
plt.ylabel('Credit Exposure')
plt.title('Counterparty Credit Risk Scenario Analysis')
plt.legend()
plt.show()

# =============================================================================
# advanced:  Let's create an example using a Monte Carlo simulation with multiple correlated risk factors. We'll consider two factors: 
    #interest rates and commodity prices. 
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# Parameters
initial_exposure = 100  # Initial exposure to the counterparty
num_scenarios = 1000   # Number of scenarios

# Define correlation matrix We use a correlation matrix to specify the correlation between interest rates and commodity prices.
correlation_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
#To clarify, a correlation coefficient of 0.5 in the matrix means that, on average, there is a positive relationship between interest rates and 
#commodity prices in the simulated scenarios, but it doesn't specify the magnitude of the relationship or imply causation. 
#The coefficients are used to generate correlated random scenarios for the Monte Carlo simulation.







# Generate random scenarios for interest rates and commodity prices
np.random.seed(42)  # for reproducibility
scenario_returns = np.random.multivariate_normal(mean=[0, 0], cov=correlation_matrix, size=num_scenarios)

# Extract interest rate and commodity price scenarios
interest_rate_scenarios = scenario_returns[:, 0]
commodity_price_scenarios = scenario_returns[:, 1]

# Simulate the evolution of interest rates and commodity prices
interest_rate_paths = np.cumprod(1 + 0.01 * interest_rate_scenarios)
commodity_price_paths = np.cumprod(1 + 0.02 * commodity_price_scenarios)

# Calculate credit exposure for each scenario
credit_exposure = initial_exposure * interest_rate_paths * commodity_price_paths

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(credit_exposure, label='Credit Exposure')
plt.axhline(y=initial_exposure, color='r', linestyle='--', label='Initial Exposure')
plt.xlabel('Scenario')
plt.ylabel('Credit Exposure')
plt.title('Counterparty Credit Risk Scenario Analysis with Multiple Factors')
plt.legend()
plt.show()

# =============================================================================
# 3 risk factors 
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# Parameters
initial_exposure = 100  # Initial exposure to the counterparty
num_scenarios = 1000   # Number of scenarios

# Define correlation matrix
correlation_matrix = np.array([[1.0, 0.6, 0.3],
                               [0.6, 1.0, 0.7],
                               [0.3, 0.7, 1.0]])

# Generate random scenarios for interest rates, commodity prices, and stock market returns
np.random.seed(42)  # for reproducibility
scenario_returns = np.random.multivariate_normal(mean=[0, 0, 0], cov=correlation_matrix, size=num_scenarios)

# Extract scenarios for interest rates, commodity prices, and stock market returns
interest_rate_scenarios = scenario_returns[:, 0]
commodity_price_scenarios = scenario_returns[:, 1]
stock_market_scenarios = scenario_returns[:, 2]

# Simulate the evolution of interest rates, commodity prices, and stock market returns
interest_rate_paths = np.cumprod(1 + 0.01 * interest_rate_scenarios)
commodity_price_paths = np.cumprod(1 + 0.02 * commodity_price_scenarios)
stock_market_paths = np.cumprod(1 + 0.015 * stock_market_scenarios)

# Calculate credit exposure for each scenario
credit_exposure = initial_exposure * interest_rate_paths * commodity_price_paths * stock_market_paths

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(credit_exposure, label='Credit Exposure')
plt.axhline(y=initial_exposure, color='r', linestyle='--', label='Initial Exposure')
plt.xlabel('Scenario')
plt.ylabel('Credit Exposure')
plt.title('Counterparty Credit Risk Scenario Analysis with Multiple Factors')
plt.legend()
plt.show()
