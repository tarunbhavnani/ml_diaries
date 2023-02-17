# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 19:12:53 2022

@author: ELECTROBOT
"""
import os
os.chdir(r'C:\Users\ELECTROBOT\Desktop\kaggle\Song_popularity')
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import logging

from sklearn import preprocessing, impute

plt.style.use('ggplot')

random_state = 42
# np.random.seed = random_state
rng = np.random.default_rng(random_state)

df = pd.read_csv('train.csv', index_col=0)
df_sub = pd.read_csv('test.csv', index_col=0)
df_sample = pd.read_csv('sample_submission.csv', index_col=0)

print(df.apply(lambda x: x.nunique()))
df.describe()

col_y = 'song_popularity'

X = df.copy()
y = X.pop(col_y)

col_cat = ['key', 'audio_mode', 'time_signature']
col_num = X.drop(columns=col_cat).columns



#eda
X_clean = X.dropna(how='any')
y_clean = y.loc[~X.isna().any(axis=1)]


fig, axs = plt.subplots(len(col_cat), figsize=(6, len(col_cat)*4))

for col, ax in zip(col_cat, axs):
    sns.histplot(X[col], ax=ax)



#q-q- plot
fig, axs = plt.subplots(len(col_num), 2,
                        figsize=(10, len(df.columns)*4))

for i, col in enumerate(col_num):
    sns.histplot(X[col], ax=axs[i, 0])
    sm.qqplot(X[col].dropna(), line="s", ax=axs[i, 1], fmt='b')
    axs[i, 1].set_title(col)

