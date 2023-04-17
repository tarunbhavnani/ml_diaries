# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 13:47:30 2023

@author: tarun
"""
import pandas as pd
import numpy as np
data= pd.read_csv(r"C:\Users\tarun\Desktop\git\ml_diaries\kaggle\WA_Fn-UseC_-Telco-Customer-Churn.csv")


hd= data.head()




hd.dtypes
nums=["MonthlyCharges", "TotalCharges"]

import re

data["TotalCharges"]=[np.nan if len(re.findall(r'\d', str(i)))==0 else float(i) for i in data["TotalCharges"]]


data["SeniorCitizen"]=["Yes" if i==1 else "No" for i in data["SeniorCitizen"]]

data['Churn']=[1 if i=="Yes" else 0 for i in data['Churn']]

[len(set(data[i])) for i in list(data)]

cats=[]

for i in data:
    if len(set(data[i]))<10:
        cats.append(i)
cats.pop()#churn

data_cats= pd.get_dummies(data[cats])
list(data_cats)

df=pd.concat([data_cats, data[nums]], axis=1)
# =============================================================================


y=data['Churn']


from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, auc,roc_curve,roc_auc_score


X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.33, random_state=411, stratify=y)


import lightgbm as lgbm  # standard alias


clf = lgbm.LGBMClassifier(objective="binary", n_estimators=1000)  # or 'mutliclass'

%time
clf.fit(X_train,y_train)

preds=clf.predict(X_test)

f1_score(y_test, preds)#.49

probs= clf.predict_proba(X_test)

probs=probs[:,1]

ns_probs = [0 for _ in range(len(y_test))]

ns_auc= roc_auc_score(y_test, ns_probs)

lgbm_auc=roc_auc_score(y_test, probs)

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lgbm_fpr, lgbm_tpr, thresholds = roc_curve(y_test, probs)

import matplotlib.pyplot as plt

# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lgbm_fpr, lgbm_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()



# =============================================================================
# 
# =============================================================================
# Import necessary libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score

df=pd.concat([data_cats, data[nums]], axis=1)
df.loc[:,"Churn"]= data["Churn"]

# Split dataset into training, validation, and test sets
train, test = train_test_split(df, test_size=0.2, random_state=42)
train, val = train_test_split(train, test_size=0.2, random_state=42)

# Define X and y variables for each set
X_train, y_train = train.drop('Churn', axis=1), train['Churn']
X_val, y_val = val.drop('Churn', axis=1), val['Churn']
X_test, y_test = test.drop('Churn', axis=1), test['Churn']

# Define objective function
objective = 'binary:logistic'

# Define hyperparameters to tune
params = {'learning_rate': [0.01, 0.05, 0.1],
          'max_depth': [3, 5, 7],
          'n_estimators': [100, 200, 500],
          'subsample': [0.5, 0.7, 1],
          'reg_alpha': [0, 0.5, 1],
          'reg_lambda': [0, 0.5, 1]}

# Create XGBoost classifier
xgb_clf = xgb.XGBClassifier(objective=objective,tree_method="hist")

# Perform grid search with cross-validation on training set
%time
grid_search = GridSearchCV(xgb_clf, param_grid=params, cv=2, scoring='roc_auc')

%%time
grid_search.fit(X_train,y_train)
best_params = grid_search.best_params_                           



# Evaluate the model on the validation set
y_val_pred = xgb.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_val_pred)
print(f'Validation AUC: {val_auc}')

# Evaluate the model on the test set
y_test_pred = xgb.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_test_pred)
print(f'Test AUC: {test_auc}')


# =============================================================================
# 
# =============================================================================
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score

# Load the dataset

# Split the dataset into training, validation, and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

# Convert the data to LightGBM Dataset format
train_set = lgb.Dataset(train_data.drop('binary_outcome', axis=1), label=train_data['binary_outcome'])
val_set = lgb.Dataset(val_data.drop('binary_outcome', axis=1), label=val_data['binary_outcome'])

# Define the LightGBM model
lgb_model = lgb.LGBMClassifier()

# Define the hyperparameters to tune
params = {
    'learning_rate': [0.1, 0.01, 0.001],
    'max_depth': [3, 5, 7],
    'num_leaves': [15, 31, 63],
    'n_estimators': [50, 100, 200],
    'subsample': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 0.5, 1],
    'reg_lambda': [0, 0.1, 0.5, 1]
}

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(lgb_model, params, cv=5, scoring='roc_auc')
grid_search.fit(train_data.drop('binary_outcome', axis=1), train_data['binary_outcome'])
best_params = grid_search.best_params_

# Train the final model with the best hyperparameters on the combined training and validation sets
lgb_model = lgb.LGBMClassifier(**best_params)
lgb_model.fit(train_data.drop('binary_outcome', axis=1).append(val_data.drop('binary_outcome', axis=1)), 
              train_data['binary_outcome'].append(val_data['binary_outcome']))

# Evaluate the model on the test set
y_test_pred = lgb_model.predict_proba(test_data.drop('binary_outcome', axis=1))[:, 1]
test_auc = roc_auc_score(test_data['binary_outcome'], y_test_pred)
print(f'Test AUC: {test_auc}')
