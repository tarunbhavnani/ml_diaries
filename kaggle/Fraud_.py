# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:20:32 2023

@author: ELECTROBOT
"""

import os
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

os.chdir(r"C:\Users\ELECTROBOT\Desktop\kaggle\Fraud")

os.listdir()

train_transaction= pd.read_csv("train_transaction.csv")
train_identity=pd.read_csv("train_identity.csv")

test_transaction= pd.read_csv("test_transaction.csv")
test_identity=pd.read_csv("test_identity.csv")


tt= train_transaction.iloc[[random.randint(0, len(train_transaction)) for i in range(1000)]]



#get cats


jk=tt.dtypes.reset_index().rename(columns={"index":"parameter", 0:"dtype"})

cat=list(jk["parameter"][jk.dtype=="object"])


# =============================================================================
# #for ordinals
# cats=[i for i in train_transaction if train_transaction[i].nunique()<10]
# =============================================================================



#na removal

class CleanData(object):
    
    def __init__(self, data):
        self.data=data
        self.cat=None
        self.ordinal=None
        self.nums=None
        self.full_data= self.missing_impute()
    
    def get_cats(self):
        
        jk=self.data.dtypes.reset_index().rename(columns={"index":"parameter", 0:"dtype"})

        self.cat=list(jk["parameter"][jk.dtype=="object"])
        
        return self.cat
    
    def get_nums(self):
        jk=self.data.dtypes.reset_index().rename(columns={"index":"parameter", 0:"dtype"})
        
        floats= list(jk["parameter"][jk["dtype"]== "float64" ])
        ints= list(jk["parameter"][jk["dtype"]== "int64" ])
        
        self.nums= floats+ints
        return self.nums
    
    def check_(self):
        nums= len(self.get_cats()+self.get_nums())
        
        assert nums==len([i for i in self.data])
    
    def get_ordinals(self,k=10):
        self.ordinal=[i for i in train_transaction if train_transaction[i].nunique()<k]
        return self.ordinal
        
        
    def missing_impute_numericals(self):
        
        nums= self.get_nums()
        data_nums= self.data[nums]
        data_nums= data_nums.fillna(data_nums.mean())
        
        return data_nums
    
    
    def missing_impute_cats(self):
        cats= self.get_cats()
        data_cats= self.data[cats]
        data_cats= data_cats.fillna(data_cats.mode())
        
        return data_cats
    
    def missing_impute(self):
        data_nums= self.missing_impute_numericals()
        data_cats= self.missing_impute_cats()
        data_cats= pd.get_dummies(data_cats)
        
        return pd.concat([data_nums, data_cats],axis = 1)
        
    
# =============================================================================
# Create data
# =============================================================================

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, auc,roc_curve,roc_auc_score

kl= CleanData(train_transaction)

data= kl.full_data

kl.check_()

X= data.drop(["isFraud"], axis=1)
y=data["isFraud"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)



# =============================================================================
# Lightgbm 
# =============================================================================


import lightgbm as lgbm  # standard alias


clf = lgbm.LGBMClassifier(objective="binary", n_estimators=1000)  # or 'mutliclass'

clf.fit(X_train,y_train)

preds=clf.predict(X_test)

f1_score(y_test, preds)

# =============================================================================
# auc
# =============================================================================
# predict probabilities
lr_probs = clf.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]

ns_probs = [0 for _ in range(len(y_test))]

# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()



# =============================================================================
# Lasso     - its for regression btw but how it works
# =============================================================================
    
from sklearn.linear_model import Lasso

# create a Lasso object with a regularization strength of 0.1
lasso = Lasso(alpha=0.1)

# fit the Lasso model to the data
lasso.fit(X_train,y_train)

# predict the target values
preds = lasso.predict(X_test)

# get the learned coefficients
coef = lasso.coef_

# identify the features that have been eliminated
eliminated_features = [i for i, c in enumerate(coef) if c == 0]



# =============================================================================
# LR
# =============================================================================

from sklearn.linear_model import LogisticRegression


lr= LogisticRegression()

lr.fit(X_train,y_train)

# predict the target values
preds=lr.predict(X_test)

f1_score(y_test, preds)

lr_probs = lr.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]

lr_auc = roc_auc_score(y_test, lr_probs)



# =============================================================================
# k fold with lgbm
# =============================================================================

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

cv = KFold(n_splits=5, random_state=1, shuffle=True)

f1=[]
for i, (train_index, test_index) in enumerate(cv.split(data)):
    print(i)
    tr= data.iloc[train_index]
    X= tr.drop(["isFraud"], axis=1)
    y=tr["isFraud"].values
    
    clf = lgbm.LGBMClassifier(objective="binary", n_estimators=1000)  # or 'mutliclass'

    clf.fit(X,y)
    
    tt= data.iloc[test_index]
    Xt= tt.drop(["isFraud"], axis=1)
    yt=tt["isFraud"].values

    
    preds=clf.predict(Xt)

    f1.append(f1_score(yt, preds))
    
    
    # [0.7366249078850405,
    #  0.7354449472096531,
    #  0.7412891986062717,
    #  0.7322404371584701,
    #  0.7256458363400202]


# =============================================================================
# #early stopping lightgbm
# =============================================================================

#X= data.drop(["isFraud"], axis=1)
#y=data["isFraud"].values

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

clf = lgbm.LGBMClassifier(objective="binary", n_estimators=10000)

eval_set = [(X_test, y_test)]

clf.fit(
    X_train,
    y_train,
    eval_set=eval_set,
    early_stopping_rounds=100,
    eval_metric="binary_logloss",
)

# predict the target values
preds=clf.predict(X_test)

f1_score(y_test, preds)

lr_probs = clf.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]

lr_auc = roc_auc_score(y_test, lr_probs)
#96.9


from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, preds)
#          0           1
#  0 array([[187835,    225],
#  1      [  2473,   4346]], dtype=int64)

#fp is less so more precision less recall


# =============================================================================
# xgboost
# =============================================================================



import xgboost as xgb
xgb_clf = xgb.XGBClassifier(
    objective="binary:logistic",
    random_state=1121218,
    n_estimators=10000,
    tree_method="hist",  # enable histogram binning in XGB
    
)

xgb_clf.fit(
    X_train,
    y_train,
    eval_set=eval_set,
    eval_metric="logloss",
    early_stopping_rounds=150,
    verbose=False,  # Disable logs
)

preds = xgb_clf.predict_proba(X_test)
lr_probs = clf.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]

lr_auc = roc_auc_score(y_test, lr_probs)
#96.9


from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, preds)