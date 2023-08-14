# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 17:03:40 2023

@author: tarun
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
    customer_dat["r"]=round((customer_dat['dt'].iloc[0]-max(customer_dat["date"])).days/30)
    customer_dat["f"]= sum(customer_dat.InvoiceNo)
    
    customer_dat["m"]=sum(customer_dat.Bill)
    
    
    return customer_dat


fd=df_.copy()
#cerate cohort data
months= set(fd.date)
final_churn_data=pd.DataFrame()
temp= fd.copy()

for start_date in sorted(months):
#    break

    cutoff_date = start_date + pd.DateOffset(months=4)
    if cutoff_date in months:
    
        dat= temp[temp.date<cutoff_date]
        dat['dt']= max(dat.date)
        
        
        #customers in this cohort of start month
        customers = dat[dat['date'] == start_date]['CustomerID']
        
        #customers in the next three months
        customers_ = dat[dat['date'] != start_date]['CustomerID']
        
        
        #all the customers who are in first month and reappear in the next three months are not churned
        nc= customers[customers.isin(customers_)]
        
        dat_= dat[dat['date'] == start_date]
        dat_=dat_.groupby("CustomerID").apply(rfm).reset_index(drop=True)
        dat_["churn"]= ~dat_["CustomerID"].isin(nc)
        #dat_["ch"]= [True if i in list(nc) else False for i in dat_['CustomerID']]
        
        
        
        final_churn_data=pd.concat([final_churn_data, dat_])
        
        #remove this month data from full data
        
        temp= temp[temp["date"]!=start_date]
    else:
        print(f"{start_date} beyond scope")







from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, auc,roc_curve,roc_auc_score, confusion_matrix

data= final_churn_data.copy()
final_churn_data.churn.value_counts()

data["churn"]=[1 if i else 0 for i in data["churn"]]

data=data[['InvoiceNo','ret_invoice','Bill','refund','Cum_bill','Cum_invoice','Cum_ret_invoice','r','f','m','Cum_refund','churn']]

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
import seaborn as sns
sns.kdeplot(probs)


f1_score(y_test, preds)
#.56

confusion_matrix(y_test, preds)

#okay precision but good recall












