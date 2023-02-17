# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 12:25:59 2022

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




plt.hist(train.iloc[:,1])
plt.hist(train.iloc[:,1], log=True)
sns.kdeplot(train.iloc[:,1])


labels= train["song_popularity"].values

train= train.drop("song_popularity", axis=1)

typ_df=train.apply(lambda x: x.nunique()).reset_index()

typ={}
cat=[i for i,j in zip(typ_df['index'],typ_df[0]) if j<20]
numr= [i for i in train if i not in cat]





train.apply(lambda x: sum(x.isnull()))

train.apply(lambda x: 100*sum(x.isnull())/len(x.isnull()))



#base model

tr= train.copy()

#fill numerical na with mean
tr[numr]=tr[numr].fillna(tr[numr].mean())

#fill cat na with most occured value
for i in cat:
    #break
    fill= tr[i].value_counts().reset_index().iloc[0][0]
    tr[i]=tr[i].fillna(fill)





#check missing 
tr.apply(lambda x: x.isna().sum() )

from sklearn.linear_model import LogisticRegression

lr= LogisticRegression()

lr.fit(tr, labels)

preds=lr.predict(tr)

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, auc,roc_curve


accuracy_score(labels,preds)

f1_score(labels,preds)

confusion_matrix(labels,preds)

#all predioted as zero

pr=[i[0] for i in lr.predict_proba(tr)]
sns.kdeplot(pr)
thresh=np.median(pr)

preds=[1 if i>thresh else 0 for i in pr]

accuracy_score(labels,preds)

f1_score(labels,preds)

confusion_matrix(labels,preds)

fpr, tpr, threshold = roc_curve(labels,preds)
roc_auc = auc(fpr, tpr)



#lasso
from sklearn import linear_model
clf = linear_model.Lasso(alpha=0.1)
clf.fit(tr,labels)
preds=clf.predict(tr)


# ss=StandardScaler()
# ss.fit(tr)
# tr=ss.transform(tr)

# for i in jk:
#     #break
#     sns.kdeplot(i)



train.describe()

label = 'song_popularity'

X = train.copy()
y = X.pop(label)

col_cat = ['key', 'audio_mode', 'time_signature']
col_num = [i for i in X.drop(columns=col_cat).columns]


#cat missing

X[col_cat].isnull().sum()

X['key']=X['key'].fillna(999)

X[col_cat].isnull().sum()







#num missing

X[col_num].isnull().sum()


from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

X_imputed=pd.DataFrame(imp_mean.fit_transform(X[col_num]), columns= col_num)
X_imputed.isnull().sum()


final_X= pd.concat([X_imputed, X[col_cat]], axis=1)



#get dummies for cats

for var in col_cat:

    final_X[var]=final_X[var].astype('str')


final_X=pd.get_dummies(final_X)


#start preictions

#train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test= train_test_split(final_X, y, test_size= .2, stratify= y, random_state=1)



#1) Log reg

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression


modellr= LogisticRegression(random_state=0).fit(X_train, Y_train)

y_pred=modellr.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score,roc_auc_score,roc_curve

accuracy_score(Y_test, y_pred)
f1_score(Y_test, y_pred) #0
confusion_matrix(Y_test, y_pred)



#Lets scale the data?

from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler = MinMaxScaler()
scaler =StandardScaler()
# transform data

scaled=pd.DataFrame(scaler.fit_transform(X_imputed), columns= col_num)

final_X= pd.concat([scaled, X[col_cat]], axis=1)

#get dummies for cats

for var in col_cat:

    final_X[var]=final_X[var].astype('str')


final_X=pd.get_dummies(final_X)

#train-test split
X_train, X_test, Y_train, Y_test= train_test_split(final_X, y, test_size= .2, stratify= y, random_state=1)

#1) Log reg


modellr= LogisticRegression(random_state=0).fit(X_train, Y_train)

y_pred=modellr.predict(X_test)

accuracy_score(Y_test, y_pred)
f1_score(Y_test, y_pred) # .4 pc
confusion_matrix(Y_test, y_pred)




from sklearn.model_selection import cross_val_score

print(cross_val_score(modellr, X_train, Y_train, cv=5,scoring="roc_auc"))

#got better bt .004?
#do we need to do some transformations to our variaboles here? Maybe. lets try Random forest and XGBosst before coming to that



# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.model_selection import cross_val_score
# from sklearn.neighbors import KNeighborsClassifier
# trans = MinMaxScaler()
# model = KNeighborsClassifier()
# pipeline = Pipeline(steps=[('t', trans), ('m', model)])
# # evaluate the pipeline
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# n_scores = cross_val_score(pipeline, X_train, Y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# # report pipeline performance
# print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))



#Randomforeest

final_X= pd.concat([X_imputed, X[col_cat]], axis=1)
final_X.describe()
#get dummies for cats

for var in col_cat:

    final_X[var]=final_X[var].astype('str')


final_X=pd.get_dummies(final_X)
X_train, X_test, Y_train, Y_test= train_test_split(final_X, y, test_size= .2, stratify= y, random_state=2)

from sklearn.ensemble import RandomForestClassifier

modelrf=RandomForestClassifier(random_state=0)
modelrf.fit(X_train, Y_train)


y_pred=modelrf.predict(X_test)

accuracy_score(Y_test, y_pred)
f1_score(Y_test, y_pred) # 1167pc
confusion_matrix(Y_test, y_pred)
#does it mean we needed transormations up in log reg? we ll soon find out!!

print(cross_val_score(modelrf, X_train, Y_train, cv=3,scoring="roc_auc"))



#lets get the best thgreshold to break and then check the accuracies

y_pred=modelrf.predict_proba(X_test)

#plot roc to get some ideas?
import scikitplot as skplt
import matplotlib.pyplot as plt

skplt.metrics.plot_roc_curve(Y_test, y_pred)
plt.show()




import scipy
from sklearn.metrics import accuracy_score

def thr_to_accuracy(thr, Y_test, predictions):
   return -accuracy_score(Y_test, np.array(predictions>thr, dtype=np.int))

best_thr = scipy.optimize.fmin(thr_to_accuracy, args=(Y_test, y_pred[:,1]), x0=0.5)

y_pred1=[0 if i[0]>best_thr[0] else 1 for i in y_pred]

confusion_matrix(Y_test, y_pred1) # 1167pc
f1_score(Y_test, y_pred1) # .23
accuracy_score(Y_test, y_pred1)
#nice improvement here but not as much as we'll lie to have



# =============================================================================
# xgboost
# =============================================================================
from xgboost.sklearn import XGBClassifier
import xgboost as xgb



xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)


#train
xgb_param = xgb1.get_xgb_params()

xgtrain = xgb.DMatrix(X_train.values, label=Y_train.values)

cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=xgb1.get_params()['n_estimators'], nfold=5,
        metrics='auc', early_stopping_rounds=10)

xgb1.set_params(n_estimators=cvresult.shape[0])


#Fit the algorithm on the data
xgb1.fit(X_train, Y_train,eval_metric='auc')


#Predict training set:
dtrain_predictions = xgb1.predict(X_train)
dtrain_predprob = xgb1.predict_proba(X_train)[:,1]


#Print model report:
print ("\nModel Report")
print ("Accuracy : %.4g" % accuracy_score(Y_train.values, dtrain_predictions))
print ("AUC Score (Train): %f" % roc_auc_score(Y_train, dtrain_predprob))

feat_imp = pd.Series(xgb1.booster().get_fscore()).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')




# =============================================================================
# box cox on parameters, since 0tive so yeo jhonson, using powertransform from sklearn
# =============================================================================

from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
#the numeric data(NA imputed)
X_imputed

for i in X_imputed:
    sm.qqplot(preprocessing.power_transform(X[[i]])[:, 0], line ='45')
    plt.show()
    

for i in X_imputed:
    
    X_imputed[i]=preprocessing.power_transform(X[[i]])[:, 0]
    
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

X_imputed=pd.DataFrame(imp_mean.fit_transform(X_imputed), columns= X_imputed.columns)




final_X= pd.concat([X_imputed, X[col_cat]], axis=1)

X_train, X_test, Y_train, Y_test= train_test_split(final_X, y, test_size= .2, stratify= y, random_state=2)





inv_sigmoid = lambda x: np.log(x / (1-x))



data_points=preprocessing.power_transform(X[['danceability']].dropna())[:,0]

sm.qqplot(data_points, line ='45')
plt.show()

#looks like an S, inv sigmoid can help here!!


data_points=preprocessing.power_transform(X[['danceability']].dropna().apply(inv_sigmoid))[:, 0]

sm.qqplot(data_points, line ='45')
plt.show()


from scipy import stats

data_points=X[['song_duration_ms']].dropna()
sm.qqplot(data_points, line ='45')
plt.show()


data_points=preprocessing.power_transform(X[['song_duration_ms']].dropna())[:,0]
sm.qqplot(data_points, line ='45')
plt.show()

plt.hist(data_points)



