# -*- coding: utf-8 -*-
"""
Created on Mon May 29 12:05:00 2023
@author: tarun+

"""

import pandas as pd

import numpy as np


data=pd.read_csv(r"C:\Users\tarun\Desktop\git\ml_diaries\kaggle\brain_size\brain_size.txt", sep=";")
data=data.iloc[:, 1:]



data.dtypes
data[data=="."]=np.nan


data.Weight=data.Weight.astype('float') 
data.Height=data.Height.astype('float') 

#improve this by alteast gender specific
data['Weight'].fillna(data['Weight'].mean(), inplace=True)
data['Height'].fillna(data['Height'].mean(), inplace=True)

data[data.Gender=="Female"].VIQ.mean()

data.groupby('Gender').mean()


import matplotlib.pyplot as plt
import seaborn as sns


sns.pairplot(data[['Gender','Weight', 'Height', 'MRI_Count']], hue= 'Gender')

from scipy import stats

female_iq= data[data.Gender=="Female"]['VIQ']
male_iq= data[data.Gender=="Male"]['VIQ']

res=stats.ttest_ind(female_iq, male_iq)

p_value=res[1]
#.444
#can not reject hypo that both are similar

# Check the p-value to determine the result
alpha = 0.05  # significance level

if p_value < alpha:
    print("The mean IQ of males is significantly diff than the mean IQ of females.")
else:
    print("There is no significant difference in the mean IQ between males and females.")


# Perform a one-sided independent t-test
t_statistic, p_value = stats.ttest_ind(male_iq, female_iq, alternative='greater')

# Check the p-value to determine the result
alpha = 0.05  # significance level

if p_value < alpha:
    print("The mean IQ of males is significantly greater than the mean IQ of females.")
else:
    print("There is no significant difference in the mean IQ between males and females.")



#lets check for weights

female_wt= data[data.Gender=="Female"]['Weight']
male_wt= data[data.Gender=="Male"]['Weight']

res=stats.ttest_ind(female_wt, male_wt)

p_value=res[1]
#.444
#can not reject hypo that both are similar



# Check the p-value to determine the result
alpha = 0.05  # significance level

if p_value < alpha:
    print("The mean weight of males is significantly diff than the mean weight of females.")
else:
    print("There is no significant difference in the mean weight between males and females.")



# Perform a one-sided independent t-test
t_statistic, p_value = stats.ttest_ind(male_wt, female_wt, alternative='greater')


if p_value < alpha:
    print("The mean weight of males is significantly greater than the mean weight of females.")
else:
    print("There is no significant difference in the mean weight between males and females.")



