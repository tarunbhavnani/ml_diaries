# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 17:19:56 2024

@author: tarun
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score,classification_report,accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")
import optuna
from optuna.samplers import TPESampler


train= pd.read_csv(r"D:\kaggle\obesity\train.csv")
test= pd.read_csv(r"D:\kaggle\obesity\test.csv")
ss= pd.read_csv(r"D:\kaggle\obesity\sample_submission.csv")



def create_new(train):
    train['senior']= [1 if i>40 else 0 for i in train['Age']]
    train['bmi']=[i/j**2 for i,j in zip(train.Weight, train.Height)]
    train=train.drop(['Weight', "Height"], axis=1)
    return train

train= create_new(train)
test= create_new(test)




def update_vars(df):
    df['FCVC'] = [1 if val < 1.5 else (2 if val < 2.5 else 3) for val in df['FCVC']]
    df['TUE'] = [0 if val < .5 else (1 if val < 1.5 else 2) for val in df['TUE']]
    df['FAF'] = [0 if val < 0.5 else (1 if val < 1.5 else (2 if val < 2.5 else 3)) for val in df['FAF']]
    df['CH2O'] = [0 if val < 0.5 else (1 if val < 1.5 else (2 if val < 2.5 else 3)) for val in df['CH2O']]
    df['NCP'] = [0 if val < 0.5 else (1 if val < 1.5 else (2 if val < 2.5 else (3 if val < 3.5 else 4))) for val in df['NCP']]
    
    return df
    



train= update_vars(train)
test= update_vars(test)




# =============================================================================


X=train.copy()
y= X.pop('NObeyesdad')


#cat_features= [i for i in X if X[i].dtype=="object"]
cat_features= [i for i in X if len(set(X[i]))<10]

df_encoded = pd.get_dummies(X, columns=cat_features)

X_train, X_test, y_train, y_test = train_test_split(df_encoded, y, test_size=0.3, random_state=1)

model = lgb.LGBMClassifier(device="gpu")

model.fit(X_train, y_train)
preds=model.predict(X_test)
f1= f1_score(y_test, preds, average= "macro")
#.87125

#lets optune

def objective(trial):
    """
    Objective function to be minimized.
    """
    param = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_class": 3,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        'device':"gpu"
    }
    gbm = lgb.LGBMClassifier(**param)
    gbm.fit(X_train, y_train)
    preds = gbm.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    return accuracy



sampler = TPESampler(seed=1)
study = optuna.create_study(study_name="lightgbm", direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=100)

print('Best parameters:', study.best_params)
#Best parameters: {'lambda_l1': 4.3036730871501596e-07, 'lambda_l2': 0.36498251380586694, 'num_leaves': 15, 
#'feature_fraction': 0.45498380010997064, 'bagging_fraction': 0.8396614832772512, 'bagging_freq': 6, 'min_child_samples': 84}



model = lgb.LGBMClassifier(**study.best_params)
model.fit(X_train, y_train)

preds1=model.predict(X_test)
f1= f1_score(y_test, preds1, average= "macro")
#.8739


# =============================================================================
# lets use k folds
# =============================================================================



from sklearn.model_selection import StratifiedKFold

folds = StratifiedKFold(n_splits=5,random_state=42,shuffle=True)
#test_preds = np.empty((num_folds, len(test)))


X_train, X_test, y_train, y_test = train_test_split(df_encoded, y, test_size=0.3, random_state=1)
f1_vals=[]
y_pred_final=0




for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_train, y_train)):
    #break
    
    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_val, y_val = X_train.iloc[valid_idx], y_train.iloc[valid_idx]
    
    
    model = lgb.LGBMClassifier(**study.best_params)
    
    
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val),verbose=300)
    
    y_pred_val = model.predict_proba(X_val)
    y_pred_max=np.argmax(y_pred_val, axis=1) 
    y_pred_proba= [i[j] for i,j in zip(y_pred_val,y_pred_max)]
    
    classes={i:num for num,i in enumerate(model.classes_)}
    #y_val.map(classes)
    f1_val = f1_score(y_val.map(classes), y_pred_max, average="weighted")
    print("f1 for fold ",n_fold,": ",f1_val)
    f1_vals.append(f1_val)
    
    y_pred_test = model.predict_proba(X_test)
    y_pred_final+=y_pred_test
    #y_pred_max=np.argmax(y_pred_test, axis=1) 
    #test_preds[n_fold, :] = y_pred_max
    print("----------------")


y_pred = np.argmax(y_pred_final, axis=1) 

classes_rev= {j:i for i,j in classes.items()}

y_pred= [classes_rev[i] for i in y_pred]

f1= f1_score(y_test, y_pred, average= "macro")
#0.8741635371490731