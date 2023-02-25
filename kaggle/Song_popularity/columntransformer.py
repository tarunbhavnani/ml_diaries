# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 15:49:31 2022

@author: ELECTROBOT
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer

import os
os.chdir(r'C:\Users\ELECTROBOT\Desktop\kaggle\Song_popularity')
from sklearn.compose import ColumnTransformer


train = pd.read_csv('train.csv', index_col=0)

list(train)

col_cat = ['key', 'audio_mode', 'time_signature']

col_num= [i for i in train if i not in col_cat]

#train.drop(columns= col_cat).columns
train.drop(columns= ['song_popularity'])



t = [('num', SimpleImputer(strategy='median'), col_num), ('cat', SimpleImputer(strategy='most_frequent'), col_cat)]

transformer = ColumnTransformer(transformers=t)

#notice how the arrangement of parameters has changes, col_num follwed by col_cat as in the pipeline
train_trans=pd.DataFrame(transformer.fit_transform(train), columns=col_num+col_cat)

#check if all good
hj= train-train_trans



#power transformer
pt= PowerTransformer()

pt = [('pow', PowerTransformer(), col_num)]
transformer = ColumnTransformer(transformers=pt,remainder='passthrough')

kl=pd.DataFrame(transformer.fit_transform(train_trans), columns=col_num+col_cat)



# =============================================================================
# put a new function using column transformer
# =============================================================================


#function transformer
from sklearn.preprocessing import FunctionTransformer

def new_fx(x):
    return x**2

transformer = FunctionTransformer(new_fx)

transformer.fit_transform(train_trans)




#use functional with columntransformer



transformer= ColumnTransformer(
    [
     ("num",FunctionTransformer(new_fx),["acousticness"])
     ],
    remainder="passthrough")

kl=transformer.fit_transform(train_trans)

kl=pd.DataFrame(kl, columns= ["acousticness"]+list(train_trans.drop(['acousticness'], axis=1)))
                


#or use it like this

def new_fx(x):
    return x**2


trans= ColumnTransformer([("new_fx", FunctionTransformer(new_fx), col_cat)], remainder='passthrough')
kl=trans.fit_transform(train)
kl=pd.DataFrame(kl, columns= col_cat+list(train.drop(col_cat, axis=1)))




#see how it react if two transformation on the same cols.basically doesnt work

trans= ColumnTransformer([("new_fx", FunctionTransformer(new_fx), col_cat),
                          ("impute", SimpleImputer(strategy='mean'), col_cat)
                          ],
                         
                         
                         remainder='passthrough')


kl=trans.fit_transform(train)
kl=pd.DataFrame(kl, columns= col_cat+col_cat+list(train.drop(col_cat, axis=1)))









