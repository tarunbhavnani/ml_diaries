# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 10:38:24 2023

@author: tarun
"""

#poisson
import numpy as np

def poisson_distribution(k, lambd):
    return (lambd ** k * np.exp(-lambd)) / np.math.factorial(k)

#2 birds singing in a minute, hwats the probability of 3 birds in the next minute


al=0
for i in range(0,10):
    al+=poisson_distribution(i,2)


from scipy.stats import poisson
import matplotlib.pyplot as plt

poisson.pmf(3, 2)

lambd=2

k_axis = np.arange(0, 25)
distribution = np.zeros(k_axis.shape[0])
for i in range(k_axis.shape[0]):
    distribution[i] = poisson.pmf(i, lambd)

plt.bar(k_axis, distribution)
# [...] Add axes, labels...

# =============================================================================
# #eg
# you know that in the past 100 hours, you received an average of 3 emails per hour, and you want to know the 
# probability of receiving 5 emails in the next hour.
# =============================================================================

import collections

# Example list of numbers
x = [1, 2, 3, 2, 1, 1, 3, 4, 2, 2, 3, 1]

# Count the number of times each value occurs
counts = collections.Counter(x)

# Calculate the total number of observations
n = len(x)

# Calculate the PMF for each value
pmf = {}
for k, count in counts.items():
    pmf[k] = count / n

# Print the PMF
for k, p in pmf.items():
    print("P(X={}) = {}".format(k, p))


pmf

import random

[random.randint(1,10) for i in range(100)]

# =============================================================================
# 
# =============================================================================

import pandas as pd
class dgp_rnd_assignment():
    """
    Data Generating Process: random assignment 
    """
    
    def generate_data(self, N=1000, seed=1):
        np.random.seed(seed)
        
        # Treatment assignment
        group = np.random.choice(['treatment', 'control'], N, p=[0.3, 0.7])
        arm_number = np.random.choice([1,2,3,4], N)
        arm = [f'arm {n}' for n in arm_number]

        # Covariates 
        gender = np.random.binomial(1, 0.5 + 0.1*(group=='treatment'), N) 
        age = np.rint(18 + np.random.beta(2 + (group=='treatment'), 5, N)*50)
        mean_income = 6 + 0.1*arm_number
        var_income = 0.2 + 0.1*(group=='treatment')
        income = np.round(np.random.lognormal(mean_income, var_income, N), 2)

        # Generate the dataframe
        df = pd.DataFrame({'Group': group, 'Arm': arm, 'Gender': gender, 'Age': age, 'Income': income})
        df.loc[df['Group']=='control', 'Arm'] = np.nan

        return df
df = dgp_rnd_assignment().generate_data()
df.head()


import seaborn as sns
sns.boxplot(data=df, x='Group', y='Income')
plt.title("Boxplot");

#the issue with the boxplot is that it hides the shape of the data, telling us some
#summary statistics but not showing us the actual data distribution.

sns.histplot(data=df, x='Income', hue='Group', bins=50)
plt.title("Histogram")



# Since the two groups have a different number of observations, the two
# histograms are not comparable
# The number of bins is arbitrary

sns.histplot(data=df, x='Income', hue='Group', bins=50,stat='density', common_norm=False)
plt.title("Density Histogram")


sns.kdeplot(df.Income[df.Group=="control"])
sns.kdeplot(df.Income[df.Group=="treatment"])


sns.kdeplot(x='Income', data=df, hue='Group', common_norm=False)
plt.title("Kernel Density Function")



#even better cummulative distrinution
sns.histplot(x='Income', data=df, hue='Group', bins=len(df),stat="density",element="step", fill=False, cumulative=True,common_norm=False)
plt.title("Cumulative distribution function")
# How should we interpret the graph?
# Since the two lines cross more or less at 0.5 (y axis), it means that their median
# is similar
# Since the orange line is above the blue line on the left and below the blue line on
# the right, it means that the distribution of the treatment group as fatter tails

#QQplot
income = df['Income'].values
income_t = df.loc[df.Group=='treatment', 'Income'].values
income_c = df.loc[df.Group=='control', 'Income'].values
df_pct = pd.DataFrame()
df_pct['q_treatment'] = np.percentile(income_t, range(100))
df_pct['q_control'] = np.percentile(income_c, range(100))
#Now we can plot the two quantile distributions against each other, plus the 45-
#degree line, representing the benchmark perfect fit.
plt.figure(figsize=(8, 8))
plt.scatter(x='q_control', y='q_treatment', data=df_pct,
label='Actual fit');
sns.lineplot(x='q_control', y='q_control', data=df_pct, color='r',
label='Line of perfect fit');
plt.xlabel('Quantile of income, control group')
plt.ylabel('Quantile of income, treatment group')
plt.legend()
plt.title("QQ plot");

#ttest

from scipy.stats import ttest_ind
stat, p_value = ttest_ind(income_c, income_t)
print(f"t-test: statistic={stat:.4f}, p-value={p_value:.4f}")

#do not reject the null hypothesis of no difference in means

#but t test depends on sample size, so some call it not reliable


#Standardized Mean Difference (SMD)


#ks
#we compute the cumulative distribution functions.

df_ks = pd.DataFrame()
income_c = df.loc[df.Group=='control', 'Income'].values
df_ks['Income'] = np.sort(df['Income'].unique())
df_ks['F_control'] = df_ks['Income'].apply(lambda x:np.mean(income_c<=x))
df_ks['F_treatment'] = df_ks['Income'].apply(lambda x:np.mean(income_t<=x))
df_ks.head()

# We now need to find the point where the absolute distance between the cumulative
# distribution functions is largest.
k = np.argmax( np.abs(df_ks['F_control'] - df_ks['F_treatment']))
ks_stat = np.abs(df_ks['F_treatment'][k] - df_ks['F_control'][k])
#visualize this
y = (df_ks['F_treatment'][k] + df_ks['F_control'][k])/2
plt.plot('Income', 'F_control', data=df_ks, label='Control')
plt.plot('Income', 'F_treatment', data=df_ks, label='Treatment')
plt.errorbar(x=df_ks['Income'][k], y=y, yerr=ks_stat/2, color='k',capsize=5, mew=3, label=f"Test statistic:{ks_stat:.4f}")
plt.legend(loc='center right');
plt.title("Kolmogorov-Smirnov Test");


#the actual test
from scipy.stats import kstest
stat, p_value = kstest(income_t, income_c)
print(f" Kolmogorov-Smirnov Test: statistic={stat:.4f}, p-value={p_value:.4f}")
#we reject


#chi sq

# Init dataframe
df_bins = pd.DataFrame()
# Generate bins from control group
_, bins = pd.qcut(income_c, q=10, retbins=True)
df_bins['bin'] = pd.cut(income_c, bins=bins).value_counts().index
# Apply bins to both groups
df_bins['income_c_observed'] = pd.cut(income_c,bins=bins).value_counts().values
df_bins['income_t_observed'] = pd.cut(income_t,bins=bins).value_counts().values
# Compute expected frequency in the treatment group
df_bins['income_t_expected'] = df_bins['income_c_observed'] /np.sum(df_bins['income_c_observed']) *np.sum(df_bins['income_t_observed'])
df_bins
from scipy.stats import chisquare
stat, p_value = chisquare(df_bins['income_t_observed'],df_bins['income_t_expected'])
print(f"Chi-squared Test: statistic={stat:.4f}, p-value={p_value:.4f}")
#we reject
