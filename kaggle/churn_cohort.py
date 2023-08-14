# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 12:35:52 2023

@author: ELECTROBOT
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

df= pd.read_csv(r"C:\Users\tarun\Desktop\MIT-Optmization\Churn\data.csv", encoding="ISO-8859-1")

df= df[~df.CustomerID.isnull()]

df["CustomerID"]=df.CustomerID.astype(int).astype(str)

df["date"]= [datetime.strptime(i.split(" ")[0], "%m/%d/%Y") for i in df.InvoiceDate]

df["month-year"] = pd.to_datetime(df['date']).dt.to_period('M')

# Convert the PeriodIndex to a DatetimeIndex with the first day of each month
df['date'] = df["month-year"].dt.to_timestamp()



#df["month-year"]=df['date'].apply(lambda x: x.strftime("%Y-%m"))
#df=df[df.Quantity>0]
df["Bill"]= df["UnitPrice"]*df["Quantity"]

df["return"]= [i if i<0 else 0 for i in df.Quantity]

df["refund"]= [i if i<0 else 0 for i in df.Bill]

df["ret_invoice"]= [i if j<0 else None for i,j in zip(df.InvoiceNo, df.Quantity)]

hd= df.sample(frac=1).head(1000)

df_=df.groupby(["CustomerID","date"]).agg({'InvoiceNo': 'nunique','ret_invoice': 'nunique', "Bill":'sum', "refund": 'sum' }).reset_index()



cumulative_spend = df_.groupby('CustomerID')['Bill'].cumsum()
df_['Cum_bill'] = cumulative_spend

cumulative_purchase = df_.groupby('CustomerID')['InvoiceNo'].cumsum()
df_['Cum_invoice'] = cumulative_purchase

cumulative_purchase = df_.groupby('CustomerID')['ret_invoice'].cumsum()
df_['Cum_ret_invoice'] = cumulative_purchase

cumulative_purchase = df_.groupby('CustomerID')['refund'].cumsum()
df_['Cum_refund'] = cumulative_purchase


def rfm(customer_dat):
    customer_dat["r"]=0
    customer_dat["f"]=0
    customer_dat["m"]=0
    
    for i in range(len(customer_dat)):
        if i>0:
            customer_dat["r"].iloc[i]=round(((customer_dat["date"].iloc[i]-customer_dat["date"].iloc[i-1]).days)/30)
        else:
            customer_dat["r"].iloc[i]=0
        customer_dat["f"].iloc[i]=customer_dat["Cum_invoice"].iloc[i]/(i+1)
        customer_dat["m"].iloc[i]=customer_dat["Cum_bill"].iloc[i]/(i+1)
    
    
    return customer_dat
    
    

# grouped= df_.groupby("CustomerID")
# fd= pd.DataFrame()
# for cid, data in grouped:
#     data= rfm(data)
#     fd= pd.concat([fd, data])

fd= df_.groupby("CustomerID").apply(rfm).reset_index(drop=True)

    



# =============================================================================
# get month wise cohorts 
# =============================================================================

months= set(fd.date)
final_churn_data=pd.DataFrame()
temp= fd.copy()

for start_date in sorted(months):
#    break

    cutoff_date = start_date + pd.DateOffset(months=4)
    if cutoff_date in months:
    
        dat= temp[temp.date<cutoff_date]
        
        #customers in this cohort of start month
        customers = dat[dat['date'] == start_date]['CustomerID']
        
        #customers in the next three months
        customers_ = dat[dat['date'] != start_date]['CustomerID']
        
        
        #all the customers who are in first month and reappear in the next three months are not churned
        nc= customers[customers.isin(customers_)]
        
        dat_= dat[dat['date'] == start_date]
        
        dat_["churn"]= ~dat_["CustomerID"].isin(nc)
        #dat_["ch"]= [True if i in list(nc) else False for i in dat_['CustomerID']]
        
        
        
        final_churn_data=pd.concat([final_churn_data, dat_])
        
        #remove this month data from full data
        
        temp= temp[temp["date"]!=start_date]
    else:
        print(f"{start_date} beyond scope")
    


# =============================================================================
# Lets put a simple xg to get the accuracy estimates
# =============================================================================

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, auc,roc_curve,roc_auc_score, confusion_matrix

data= final_churn_data.copy()
final_churn_data.churn.value_counts()

data["churn"]=[1 if i else 0 for i in data["churn"]]
hd= data.head()

list(data)

data=data[['InvoiceNo','ret_invoice','Bill','refund','Cum_bill','Cum_invoice','Cum_ret_invoice','Cum_refund','r','f','m','churn']]


X= data.drop(["churn"], axis=1)
y=data["churn"].values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=411, stratify=y)


import lightgbm as lgbm  # standard alias




clf = lgbm.LGBMClassifier(objective="binary", n_estimators=1000)  # or 'mutliclass'

clf.fit(X_train,y_train)

preds=clf.predict(X_test)

f1_score(y_test, preds)#.49
confusion_matrix(y_test, preds)


probs= clf.predict_proba(X_test)

probs=probs[:,1]

ns_probs = [0 for _ in range(len(y_test))]

ns_auc= roc_auc_score(y_test, ns_probs)

lgbm_auc=roc_auc_score(y_test, probs)

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lgbm_fpr, lgbm_tpr, thresholds = roc_curve(y_test, probs)

# idx = np.argmax(lgbm_tpr - lgbm_fpr)
# best_threshold = thresholds[idx]
# print("Best threshold:", best_threshold)

#def best_thresh():
best_f1=0
best_thresh=0
probs= clf.predict_proba(X_test)
probs=probs[:,1]
for thresh in range(0,100):
    thresh= thresh/100    
    preds=[1 if i>thresh else 0 for i in probs]
    f1= f1_score(y_test, preds)
    if f1>best_f1:
        print(f1)
        best_thresh=thresh
        best_f1=f1


preds=[1 if i>best_thresh else 0 for i in probs]
sns.kdeplot(probs)

confusion_matrix(y_test, preds)


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


#lets see ks

from scipy.stats import ks_2samp
predicted_probabilities = [p for _, p in sorted(zip(probs, y_test), reverse=True)]
ks_statistic, p_value = ks_2samp(predicted_probabilities, true_values)


# =============================================================================
# grid search lgbm
# =============================================================================

model = lgbm.LGBMClassifier(objective="binary", n_estimators=1000)  # or 'mutliclass'
# Define the hyperparameter grid for GridSearchCV
param_grid = {
    'learning_rate': [0.1, 0.05, 0.01],
    'num_leaves': [31, 63, 127],
    'max_depth': [-1, 5, 10],
    'min_child_samples': [20, 50, 100],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# Perform grid search
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)  # X_train and y_train are your training data

# Print the best parameters and best score
print("Best Parameters: ", grid_search.best_params_)
#Best Parameters:  {'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 5, 'min_child_samples': 20, 'num_leaves': 63, 'subsample': 0.8}
print("Best Score: ", grid_search.best_score_)

best_params = grid_search.best_params_

# Create a new LightGBM model with the best parameters
model = lgbm.LGBMClassifier(objective="binary", n_estimators=1000)
model.set_params(**best_params)

# Train the model on your training data
model.fit(X_train, y_train)

preds=model.predict(X_test)

f1_score(y_test, preds)#.49
confusion_matrix(y_test, preds)

#def best_thresh():
best_f1=0
best_thresh=0
probs= model.predict_proba(X_test)
probs=probs[:,1]
for thresh in range(0,100):
    thresh= thresh/100    
    preds=[1 if i>thresh else 0 for i in probs]
    f1= f1_score(y_test, preds)
    if f1>best_f1:
        print(f1)
        best_thresh=thresh
        best_f1=f1


preds=[1 if i>best_thresh else 0 for i in probs]

f1_score(y_test, preds)#.61
confusion_matrix(y_test, preds)

# =============================================================================
# xg
# =============================================================================


import xgboost as xgb
xgb_clf = xgb.XGBClassifier(
    objective="binary:logistic",
    random_state=1121218,
    n_estimators=10000,
    tree_method="hist",  # enable histogram binning in XGB
    
)

eval_set = [(X_test, y_test)]

%time
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
#75.5

#def best_thresh():
best_f1=0
best_thresh=0
probs= clf.predict_proba(X_test)
probs=probs[:,1]
for thresh in range(0,100):
    thresh= thresh/100    
    preds=[1 if i>thresh else 0 for i in probs]
    f1= f1_score(y_test, preds)
    if f1>best_f1:
        best_thresh=thresh
        best_f1=f1




from sklearn.metrics import confusion_matrix
probs = clf.predict_proba(X_test)
preds= [1 if i>best_thresh else 0 for i in probs[:,1]]
f1_score(y_test, preds)
confusion_matrix(y_test, preds)

tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

#importnant features
importance_dict = xgb_clf.get_booster().get_score(importance_type="weight")
print("Feature Importances:")
for feature, importance in importance_dict.items():
    print("%s: %f" % (feature, importance))


#itr seems that the bigger the numerical value of the feature the more the importance

# =============================================================================
# 
# =============================================================================
from sklearn.preprocessing import StandardScaler
data=data[['InvoiceNo','ret_invoice','Bill','refund','Cum_bill','Cum_invoice','Cum_ret_invoice','Cum_refund','r','f','m','churn']]
X= data.drop(["churn"], axis=1)
y=data["churn"].values

feature_names = list(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=411, stratify=y)


xgb_clf = xgb.XGBClassifier(
    objective="binary:logistic",
    random_state=1121218,
    n_estimators=10000,
    tree_method="hist",  # enable histogram binning in XGB
    
)

eval_set = [(X_test, y_test)]

xgb_clf.fit(
    X_train,
    y_train,
    eval_set=eval_set,
    eval_metric="logloss",
    early_stopping_rounds=150,
    verbose=False,  # Disable logs
)

preds = xgb_clf.predict_proba(X_test)
lr_probs = xgb_clf.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]

lr_auc = roc_auc_score(y_test, lr_probs)
#75.5

#def best_thresh():
best_f1=0
best_thresh=0
probs= xgb_clf.predict_proba(X_test)
probs=probs[:,1]
for thresh in range(0,100):
    thresh= thresh/100    
    preds=[1 if i>thresh else 0 for i in probs]
    f1= f1_score(y_test, preds)
    if f1>best_f1:
        best_thresh=thresh
        best_f1=f1


probs = xgb_clf.predict_proba(X_test)
preds= [1 if i>best_thresh else 0 for i in probs[:,1]]
f1_score(y_test, preds)
confusion_matrix(y_test, preds)

tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

#importnant features
importance_dict = xgb_clf.get_booster().get_score(importance_type="weight")
print("Feature Importances:")
for a,b in zip(importance_dict.items(),feature_names):
    
    print("%s: %f" % (b, a[1]))
# =============================================================================
# scaling didnt change much
# =============================================================================


