# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 13:58:07 2024

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
from catboost import CatBoostClassifier, Pool
from catboost.utils import eval_metric
import warnings
warnings.filterwarnings("ignore")
import optuna
from optuna.samplers import TPESampler


train= pd.read_csv(r"D:\kaggle\obesity\train.csv")
test= pd.read_csv(r"D:\kaggle\obesity\test.csv")
ss= pd.read_csv(r"D:\kaggle\obesity\sample_submission.csv")


list(train)
train.NObeyesdad.value_counts()

def create_new(train):
    train['senior']= [1 if i>40 else 0 for i in train['Age']]
    train['bmi']=[i/j**2 for i,j in zip(train.Weight, train.Height)]
    train=train.drop(['Weight', "Height"], axis=1)
    return train

train= create_new(train)
test= create_new(test)

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



# eating habbits
# The attributes related with eating habits are: 
#     Frequent consumption of high caloric food (FAVC),
#     Frequency of consumption of vegetables (FCVC),
#     Number of main meals (NCP),
#     Consumption of food between meals (CAEC),
#     Consumption of water daily (CH20),
#     and Consumption of alcohol (CALC).
    
# =============================================================================
# univariate
# =============================================================================




train[['FAVC','NObeyesdad' ]]


sns.countplot(data= train, x= 'FAVC', hue= 'NObeyesdad')

#no obs level 3 in no, lets cerate a list oifr this
no_obs_3={'FAVC':"yes"}




train[train.FAVC=='no']['NObeyesdad'].value_counts()
#no Obesity_Type_III, minimmum Obesity_Type_II and Obesity_Type_I



#train['FCVC'] = [1 if val < 1.5 else (2 if val < 2.5 else 3) for val in train['FCVC']]
sns.countplot(data= train, x= 'FCVC', hue= 'NObeyesdad')
no_obs_3['FCVC']=3


#train['NCP'] = [0 if val < 0.5 else (1 if val < 1.5 else (2 if val < 2.5 else (3 if val < 3.5 else 4))) for val in train['NCP']]
sns.countplot(data= train, x= 'NCP', hue= 'NObeyesdad')
no_obs_3['NCP']=3

sns.countplot(data= train, x= 'CH2O', hue= 'NObeyesdad')
sns.countplot(data= train, x= 'CAEC', hue= 'NObeyesdad')
no_obs_3['CAEC']='Sometimes'

sns.countplot(data= train, x= 'CALC', hue= 'NObeyesdad')
no_obs_3['CALC']='Sometimes'

train.boxplot(column=['bmi'], by='NObeyesdad')




# =============================================================================
# #lets try clustering and visualiztion
# =============================================================================


data= train[['FAVC','FCVC', 'NCP', 'CAEC', 'CH2O', 'CALC']]


from kmodes.kprototypes import KPrototypes

# Load your data

# Separate continuous and categorical variables
continuous_cols = [i for i in data if data[i].dtypes!=object]
categorical_cols = [i for i in data if data[i].dtypes==object]

# Initialize K-Prototypes model
kproto = KPrototypes(n_clusters=3, verbose=2)  # You can adjust the number of clusters as needed

# Fit the model
clusters = kproto.fit_predict(data.values, categorical=[0,3,5])  # Replace [5, 6] with the indices of categorical columns

# Add cluster labels to the original dataframe
data['cluster'] = clusters
data.cluster.value_counts()

data['NObeyesdad']=train['NObeyesdad']

data.groupby('NObeyesdad')['cluster'].value_counts()



# Visualize clusters
# Pairplot for continuous variables
sns.pairplot(data=data, vars=continuous_cols, hue='cluster')
plt.title('Pairplot of Continuous Variables')
plt.show()

# Countplot for categorical variables
for col in categorical_cols:
    plt.figure(figsize=(8, 6))
    sns.countplot(data=data, x=col, hue='cluster')
    plt.title(f'Countplot of {col} by Cluster')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()




# The attributes related with the physical condition are: 
#     Calories consumption monitoring (SCC), 
#     Physical activity frequency (FAF), 
#     Time using technology devices (TUE), 
#     Transportation used (MTRANS)

sns.countplot(data= train, x= 'SCC', hue= 'NObeyesdad')
no_obs_3['SCC']='no'

sns.countplot(data= train, x= 'FAF', hue= 'NObeyesdad')
sns.countplot(data= train, x= 'TUE', hue= 'NObeyesdad')
sns.countplot(data= train, x= 'MTRANS', hue= 'NObeyesdad')

train[train['MTRANS']!="Public_Transportation"]['NObeyesdad'].value_counts()



# =============================================================================
# optuna cat
# =============================================================================





X=train.copy()
y= X.pop('NObeyesdad')

cat_features= [i for i in X if len(set(X[i]))<10]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


model = CatBoostClassifier(
eval_metric='AUC',
learning_rate=0.03,
iterations=100,
cat_features=cat_features,
task_type='GPU' )

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
f1_score(y_test, y_pred, average='macro')
#.8383
#optuna
def objective(trial):
    model = CatBoostClassifier(
        iterations=trial.suggest_int("iterations", 100, 500),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        depth=trial.suggest_int("depth", 4, 10),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-8, 100.0, log=True),
        bootstrap_type=trial.suggest_categorical("bootstrap_type", ["Bayesian"]),
        random_strength=trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 10.0),
        od_type=trial.suggest_categorical("od_type", ["IncToDec", "Iter"]),
        od_wait=trial.suggest_int("od_wait", 10, 50),
        verbose=True,task_type='GPU',cat_features=cat_features
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

optuna.logging.set_verbosity(optuna.logging.WARNING)

sampler = TPESampler(seed=1)
study = optuna.create_study(study_name="catboost", direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=100)

trial = study.best_trial
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
# iterations: 326
# learning_rate: 0.06211003994893942
# depth: 5
# l2_leaf_reg: 0.028070723716320668
# bootstrap_type: Bayesian
# random_strength: 2.6110735398324655e-08
# bagging_temperature: 0.20366464111245872
# od_type: Iter
# od_wait: 27

#create model
model = CatBoostClassifier(**trial.params, verbose=True,cat_features=cat_features)
model.fit(X_train, y_train)
y_pred1 = model.predict(X_test)
print(classification_report(y_test, y_pred1))
f1_score(y_test, y_pred1, average='macro')
#.87

# =============================================================================
# next we do prunning, with lightgbm
# =============================================================================



