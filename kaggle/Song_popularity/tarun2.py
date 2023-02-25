# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 12:48:11 2022

@author: ELECTROBOT
"""

import os
os.chdir(r'C:\Users\ELECTROBOT\Desktop\kaggle\Song_popularity')
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
# import statsmodels.api as sm
# import logging

from sklearn import preprocessing, impute

plt.style.use('ggplot')

random_state = 42
# np.random.seed = random_state
#rng = np.random.default_rng(random_state)


train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)
sample = pd.read_csv('sample_submission.csv', index_col=0)

train.head()

train.apply(lambda x: sum(x.isnull()))

labels= train.pop("song_popularity")

cat= [i for i in train if train[i].nunique()<20]

num= [i for i in train if i not in cat]


train[num]=train[num].fillna(train[num].mean())

for i in cat:
    train[i]= train[i].astype("category")

train[cat]=train[cat].apply(lambda x: x.fillna(x.value_counts().index[0]))

train.apply(lambda x: sum(x.isnull()))


#lets apply log reg

from sklearn.linear_model import LogisticRegression

clf= LogisticRegression()
clf.fit(train,labels)

preds=clf.predict(train)

#all zero


train_=pd.get_dummies(train[cat], drop_first=True)

train_final= pd.concat([train[num], train_], axis=1)


clf= LogisticRegression()
clf.fit(train_final,labels)

preds=clf.predict(train_final)
sum(preds)

#all zero
preds=clf.predict_proba(train_final)
sns.kdeplot([i[0] for i in preds])
thresh=np.median([i[0] for i in preds])

pred=[0 if i[0]>=thresh else 1 for i in preds]

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
accuracy_score(labels, pred)
f1_score(labels, pred)


#not good, full bias

from sklearn.preprocessing import StandardScaler

std= StandardScaler()
std.fit(train[num])
train_1=pd.DataFrame(std.transform(train[num]), columns= list(train[num]))

train_final= pd.concat([train_1, train_], axis=1)

clf= LogisticRegression()
clf.fit(train_final,labels)

preds=clf.predict(train_final)
sum(preds)

#all zero
preds=clf.predict_proba(train_final)
sns.kdeplot([i[0] for i in preds])
thresh=np.median([i[0] for i in preds])

pred=[0 if i[0]>=thresh else 1 for i in preds]

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
accuracy_score(labels, pred)
f1_score(labels, pred)

#lil better but not there, big bias


from sklearn.svm import SVC
clf = SVC(gamma='auto',probability=True)

clf.fit(train_final,labels)

preds=clf.predict(train_final)

sum(preds)

#all zero
preds=clf.predict_proba(train_final)
sns.kdeplot([i[0] for i in preds])
thresh=np.median([i[0] for i in preds])

pred=[0 if i[0]>=thresh else 1 for i in preds]

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
accuracy_score(labels, pred)#63pc
f1_score(labels, pred)#57pc

#still better but not there, big bias





#linear svc
from sklearn.svm import LinearSVC
clf=LinearSVC(random_state=0, tol=1e-05)

clf.fit(train_final,labels)

preds=clf.predict(train_final)

sum(preds)

#all zero
preds=clf.predict_proba(train_final)
sns.kdeplot([i[0] for i in preds])
thresh=np.median([i[0] for i in preds])

pred=[0 if i[0]>=thresh else 1 for i in preds]

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
accuracy_score(labels, pred)
f1_score(labels, pred)

#lil better but not there, big bias




#xg
import xgboost as xgb
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
xgb_model.fit(train_final,labels)


preds=xgb_model.predict(train_final)

sum(preds)

#all zero
preds=xgb_model.predict_proba(train_final)
sns.kdeplot([i[0] for i in preds])
thresh=np.median([i[0] for i in preds])

pred=[0 if i[0]>=thresh else 1 for i in preds]
sum(pred)
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
accuracy_score(labels, pred)#73
f1_score(labels, pred)#691

#wallah, improved with almost no hassle


from sklearn.metrics import roc_auc_score

def get_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

get_auc(labels, pred)


















