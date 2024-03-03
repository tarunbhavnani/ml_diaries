# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 14:13:20 2024

@author: tarun
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from catboost import CatBoostClassifier, Pool
from catboost.utils import eval_metric
import warnings
warnings.filterwarnings("ignore")


train= pd.read_csv(r"D:\kaggle\obesity\train.csv")
test= pd.read_csv(r"D:\kaggle\obesity\test.csv")
ss= pd.read_csv(r"D:\kaggle\obesity\sample_submission.csv")


list(train)
train.NObeyesdad.value_counts()




train.head()
train.describe()

[(i,len(set(train[i]))) for i in train]
#[(i,len(set(train[i]))) for i in train if train[i].dtype=='object']

cats=[i for i in train if train[i].dtype=='object']
print("cats:",cats)
nums=[i for i in train if train[i].dtype!='object']
print("nums:",nums)


fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))

# Plot density plots for each column
count=[(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
for num, name in zip(count,nums):
    sns.kdeplot(data=train[name], ax=axes[num], fill=True)

    # Set titles for each subplot
    axes[num].set_title(f'{name}')

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()



def update_vars(df):
    df['FCVC'] = [1 if val < 1.5 else (2 if val < 2.5 else 3) for val in df['FCVC']]
    df['TUE'] = [0 if val < .5 else (1 if val < 1.5 else 2) for val in df['TUE']]
    df['FAF'] = [0 if val < 0.5 else (1 if val < 1.5 else (2 if val < 2.5 else 3)) for val in df['FAF']]
    df['CH2O'] = [0 if val < 0.5 else (1 if val < 1.5 else (2 if val < 2.5 else 3)) for val in df['CH2O']]
    df['NCP'] = [0 if val < 0.5 else (1 if val < 1.5 else (2 if val < 2.5 else (3 if val < 3.5 else 4))) for val in df['NCP']]
    
    return df
    



train= update_vars(train)
test= update_vars(test)




def create_new(train):
    train['senior']= [1 if i>40 else 0 for i in train['Age']]
    train['bmi']=[i/j**2 for i,j in zip(train.Weight, train.Height)]
    return train


train= create_new(train)
test= create_new(test)


features= [i for i in train if i not in ["id","NObeyesdad"]]
print(features)


cat_features= [i for i in features if len(set(train[i]))<20]
print(cat_features)

RAND_VAL=42
#num_folds=3 ## Number of folds
n_est=1000 ## Number of estimators
X=train[features]
y=train['NObeyesdad']
 
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier, Pool
folds = StratifiedKFold(n_splits=5,random_state=42,shuffle=True)
#test_preds = np.empty((num_folds, len(test)))
f1_vals=[]
y_pred_final=0

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[valid_idx], y.iloc[valid_idx]
    
    train_pool = Pool(X_train, y_train,cat_features=cat_features)
    val_pool = Pool(X_val, y_val,cat_features=cat_features)
    
    clf = CatBoostClassifier(
    eval_metric='AUC',
    learning_rate=0.03,
    iterations=n_est,
    task_type='GPU' )
    clf.fit(train_pool, eval_set=val_pool,verbose=300)
    
    y_pred_val = clf.predict_proba(X_val[list(X)])
    y_pred_max=np.argmax(y_pred_val, axis=1) 
    y_pred_proba= [i[j] for i,j in zip(y_pred_val,y_pred_max)]
    
    classes={i:num for num,i in enumerate(clf.classes_)}
    #y_val.map(classes)
    f1_val = f1_score(y_val.map(classes), y_pred_max, average="weighted")
    print("f1 for fold ",n_fold,": ",f1_val)
    f1_vals.append(f1_val)
    
    y_pred_test = clf.predict_proba(test[features])
    y_pred_final+=y_pred_test
    #y_pred_max=np.argmax(y_pred_test, axis=1) 
    #test_preds[n_fold, :] = y_pred_max
    print("----------------")



y_pred = np.argmax(y_pred_final, axis=1) 

classes_rev= {j:i for i,j in classes.items()}

test['NObeyesdad']= [classes_rev[i] for i in y_pred]

sub= test[['id','NObeyesdad' ]]

sub.to_csv("submission.csv",index=False)

# #f1 values :
#     [0.9040666961345803,
#      0.8982184954091234,
#      0.9048216526949133,
#      0.8974882067579881,
#      0.8987990723827118]

# =============================================================================
# lets improve catboost model above, get better metric than f1

# categorical variables are to be explored for some feature engg

# next would be ensemble with xg and random


# also we can create a few more variables
# =============================================================================



train[cat_features]

[train[i].value_counts() for i in train[cat_features]]


from collections import Counter
for feat in cat_features:
   
    #feat="FAVC"
    print(train.groupby(feat)['NObeyesdad'].apply(lambda x: Counter(x)).reset_index())
    
    
# =============================================================================
# lets see what we are predicting
# =============================================================================


train['NObeyesdad'].value_counts(normalize=True)
#distribution is not bad




