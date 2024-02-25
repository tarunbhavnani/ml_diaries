# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 18:30:46 2024

@author: tarun
"""

import os
os.chdir(r'D:\kaggle\bank churn dataset')
import pandas as pd

train= pd.read_csv('train.csv')
test= pd.read_csv('test.csv')
ss= pd.read_csv('sample_submission.csv')


train.Exited.value_counts(normalize=True)

list(train)


[len(set(train[i])) for i in train]

kl=train.groupby(['CustomerId', 'Geography'])['Exited'].sum().reset_index()

train.dtypes
#check na
train.isna().sum()

#drop redundant variables
drop= ['id', "CustomerId", "Surname"]

train=train.drop(drop, axis=1)



#encode variables

gender={"Male":1, "Female":0}
train["Gender"]=train.Gender.map(gender)



#pd.get_dummies(train['Geography'])
train = pd.get_dummies(train, columns=['Geography'], prefix='country')
train = train.astype(float)



# =============================================================================
# #lets train a simple xgboost
# =============================================================================

import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score

y= train.Exited
X= train.drop("Exited", axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost classifier
#model = XGBClassifier(objective='binary:logistic', random_state=42)
model = XGBClassifier(objective='binary:logistic',subsample=0.8, colsample_bytree=0.8, colsample_bylevel=0.8,n_estimators=500, random_state=42)

# Set up cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Perform cross-validated training with early stopping
cv_results = cross_val_score(
    model,
    X_train, y_train,
    cv=cv,
    scoring='roc_auc',  # Use AUC as the scoring metric
    fit_params={
        'early_stopping_rounds': 10,
        'eval_metric': 'auc',
        'eval_set': [(X_train, y_train)],
        'verbose': False
    }
)

# Display the cross-validated AUC scores
print("Cross-validated AUC scores:", cv_results)
print("Mean AUC:", cv_results.mean())

# Train the final model on the full training set
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluate the final model using AUC
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"Final AUC Score: {auc_score}")

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba[:,1])
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Display the results
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)

# =============================================================================
# 
# =============================================================================


pd.get_dummies(train['Geography'])
df_encoded = pd.get_dummies(train, columns=['Geography'], prefix='country')
df_encoded = df_encoded.astype(int)

y= df_encoded.Exited
X= df_encoded.drop("Exited", axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid to search
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
}

# Initialize XGBoost classifier
xgb_model = XGBClassifier(objective='binary:logistic', random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=3)

# Perform grid search and cross-validation
grid_search.fit(X_train, y_train)

# Display the best hyperparameters
best_params = grid_search.best_params_
# {'colsample_bytree': 0.9,
#  'learning_rate': 0.1,
#  'max_depth': 4,
#  'n_estimators': 200,
#  'subsample': 1.0}
print(f"Best Hyperparameters: {best_params}")

# Make predictions on the test set using the best model
y_pred = grid_search.best_estimator_.predict(X_test)
y_proba = grid_search.best_estimator_.predict_proba(X_test)

# Evaluate the model
accuracy = f1(y_test, y_pred)
print(f"Accuracy: {accuracy}")
auc = roc_auc_score(y_test, y_proba[:,1])


# =============================================================================
# 
# =============================================================================

xgb_model = XGBClassifier(objective='binary:logistic', colsample_bytree= 0.9,
  learning_rate= 0.1,
  max_depth= 4,
  n_estimators= 200,
  subsample= 1.0)

model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba[:,1])
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
