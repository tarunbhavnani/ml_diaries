# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 17:12:15 2022

@author: ELECTROBOT
"""

import os
os.chdir(r'C:\Users\ELECTROBOT\Desktop\kaggle\Song_popularity')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

train= pd.read_csv('train.csv')

#independent
train['song_popularity'].value_counts()

#missing
[(i,sum(train[i].isnull())) for i in train]


# plt.hist(train['liveness'])
# list(train)




#number of features

[(i,len(set(train[i]))) for i in train]

train['key']=train['key'].astype('object')
train['audio_mode']=train['audio_mode'].astype('object')
train['time_signature']=train['time_signature'].astype('object')
train['song_popularity']=train['song_popularity'].astype('object')

train.dtypes


#plots
sns.set_context("paper", font_scale=3)    
# sns.distplot(train['acousticness'], hist=False)

# fig, ax = plt.subplots(nrows=3)
# sns.distplot(train['liveness'], hist=False)
# sns.distplot(train['liveness'], hist=False, color="Red", rug="True")
# sns.distplot(train['acousticness'], hist=False)
# plt.show()


f = plt.figure()
for num,i in enumerate(list(train)):
    #break
    f.add_subplot(5, 3, num+1)
    sns.distplot(train[i], hist=False)
    plt.show()


fig, axes = plt.subplots(nrows=3, ncols=3,figsize=(24, 10))
palette_ = ["#9b59b6", "#ff0000", "#00f0f0", "#00ff00"]
sns.kdeplot(train["song_duration_ms"], fill=True, color="blue", alpha=1, ax = axes[0,0])
sns.kdeplot(train["acousticness"], fill=True, color="red", alpha=1,  ax = axes[0,1])
sns.kdeplot(train["danceability"], fill=True, color="violet", alpha=1,  ax = axes[0,2])
sns.kdeplot(train["instrumentalness"], fill=True, color="orange", alpha=1,  ax = axes[1,0])
sns.kdeplot(train["liveness"], fill=True, color="yellow", alpha=1,  ax = axes[1,1])
sns.kdeplot(train["loudness"], fill=True, color="green", alpha=1, ax = axes[1,2])
sns.kdeplot(train["speechiness"], fill=True, color="pink", alpha=1,  ax = axes[2,0])
sns.kdeplot(train["tempo"], fill=True, color="lightblue", alpha=1,  ax = axes[2,1])
sns.kdeplot(train["audio_valence"], fill=True, color="lightgreen", alpha=1,  ax = axes[2,2])
fig.tight_layout()
fig.show()


#target impact

f = plt.figure()
for num,i in enumerate(list(train)):
    #break
    f.add_subplot(5, 3, num+1)
    sns.distplot(train[i][train[i].notnull()][train['song_popularity']==1], hist=False)
    sns.distplot(train[i][train[i].notnull()][train['song_popularity']==0], hist=False)
    plt.show()

#if skewed
#sns.distplot(train[i][train[i].notnull()][train['song_popularity']==1].apply(lambda x: np.log(x)))


#correlation

plt.rcParams["figure.figsize"] = (18,12)
dataplot = sns.heatmap(train.corr(), cmap="viridis", annot=True)
plt.show()

corr= train.corr()

#check all where correlation>.5
#+ve
[i for i in itertools.product(list(corr), list(corr)) if corr.loc[i[0],i[1]]>.5 and corr.loc[i[0],i[1]]!=1]

plt.scatter(train['energy'], train['loudness'])
#-ve
[i for i in itertools.product(list(corr), list(corr)) if corr.loc[i[0],i[1]]<-.5 and corr.loc[i[0],i[1]]!=1]
plt.scatter(train['energy'], train['acousticness'])
plt.scatter(train['acousticness'], train['loudness'])


#with target
plt.scatter(train['acousticness'], train['loudness'], c= train['song_popularity'])
plt.scatter(train['energy'], train['loudness'], c= train['song_popularity'])
#missing values by regression for continous variables


list(train)

variables=['song_duration_ms','acousticness','danceability','energy','instrumentalness','liveness',
            'loudness','speechiness','tempo','audio_valence']

train_req= train[variables]

for var in variables:
    #print(var)
    
    
    if sum(train[var].isnull())>0:
        #break
        print(var)
        train_temp= train_req[train_req[var].isnull()==False]
        test_temp= train_req[train_req[var].isnull()==True]
        
        
        #y= train_temp.pop(var)
        y= train_temp[var]
        
        
        #impute the rest nulls with mean
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        
        imp.fit(train_temp.loc[:, train_temp.columns != var].drop_duplicates())
        
        ty=imp.transform(train_temp.loc[:, train_temp.columns != var])
        
        
        
        regr = RandomForestRegressor(max_depth=6, random_state=0)
        regr.fit(ty, y)
        
        #test_temp.pop(var)
        
        tt=imp.transform(test_temp.loc[:, test_temp.columns != var])
        
        
        test_temp[var]=regr.predict(tt)
        
        
        train_req= train_temp.append(test_temp)
        

        
train[variables]=train_req


        
#we have done for only continous, how to include categorical here!
        
        




# =============================================================================
# direct implementation of lgbm
# =============================================================================












