
import os
os.chdir(r'D:\kaggle\bank churn dataset')
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler


train= pd.read_csv('train.csv')
test= pd.read_csv('test.csv')
ss= pd.read_csv('sample_submission.csv')



surname=train.groupby("Surname")['Exited'].sum().to_dict()

train['Surname']=train['Surname'].map(surname)
train['Surname']=pd.cut(train['Surname'], bins=20, labels=list(range(1, 21)))


test['Surname']=[surname[i] if i in surname else 0  for i in test["Surname"]]
test['Surname']=pd.cut(test['Surname'], bins=20, labels=list(range(1, 21)))


train['CreditScore']=pd.cut(train['CreditScore'], bins=5, labels=list(range(1, 6)))


test['CreditScore']=pd.cut(test['CreditScore'], bins=5, labels=list(range(1, 6)))


train['Age']=pd.cut(train['Age'], bins=10, labels=list(range(1, 11)))


test['Age']=pd.cut(test['Age'], bins=10, labels=list(range(1, 11)))


train['Balance']=pd.cut(train['Balance'], bins=20, labels=list(range(1, 21)))

test['Balance']=pd.cut(test['Balance'], bins=20, labels=list(range(1, 21)))


train['EstimatedSalary']=pd.cut(train['EstimatedSalary'], bins=20, labels=list(range(1, 21)))

test['EstimatedSalary']=pd.cut(test['EstimatedSalary'], bins=20, labels=list(range(1, 21)))


# import seaborn as sns
# sns.kdeplot(train.salary_normalized)
# pd.cut(train['EstimatedSalary'], bins=20)
# #train.Exited.value_counts(normalize=True)
# scaler = MinMaxScaler()
# train['EstimatedSalary'] = scaler.fit_transform(train[['EstimatedSalary']])
# #list(train)


#[len(set(train[i])) for i in train]

#kl=train.groupby(['CustomerId', 'Geography'])['Exited'].sum().reset_index()

#train.dtypes
#check na
#train.isna().sum()

#drop redundant variables
#drop= ['id', "CustomerId", "Surname"]
drop= ['id', "CustomerId"]

train=train.drop(drop, axis=1)



#encode variables

gender={"Male":1, "Female":0}
train["Gender"]=train.Gender.map(gender)



#pd.get_dummies(train['Geography'])
train = pd.get_dummies(train, columns=['Geography'], prefix='country')
train = train.astype(float)

train=train.drop_duplicates()


y= train.Exited
X= train.drop("Exited", axis=1)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define XGBoost parameters with scale_pos_weight
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],
    'subsample': 0.9,
    'max_depth':4,
    'colsample_bytree': 0.9,
    'colsample_bylevel': 0.9,
    'scale_pos_weight': 3.5,  # Adjust this value based on the imbalance in your data
    'random_state': 91,
}

# Initialize XGBoost classifier with early stopping
model = XGBClassifier(**xgb_params)

# Train the model with early stopping
evals = [(X_train, y_train), (X_test, y_test)]  # Validation set(s)
model.fit(
    X_train, y_train,
    eval_set=evals,
    early_stopping_rounds=10,  # Stop if no improvement for 10 rounds
    verbose=True
)

# Make predictions on the test set
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluate the model using AUC
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"AUC Score: {auc_score}")
# if auc_score>mx_auc:
#     mx_auc=auc_score
#     best=i
#     #89

#random_state=91


# =============================================================================
# testing
# =============================================================================

test1=test.drop(drop, axis=1)
test1["Gender"]=test1.Gender.map(gender)
test1 = pd.get_dummies(test1, columns=['Geography'], prefix='country')
test1 = test1.astype(float)

test_proba = model.predict_proba(test1)[:, 1]

test['Exited']=test_proba
final= test[['id', 'Exited']]




# =============================================================================
# 
# =============================================================================

from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier, Pool
folds = StratifiedKFold(n_splits=5,random_state=42,shuffle=True)
#test_preds = np.empty((num_folds, len(df_test)))
auc_vals=[]
n_est=3500 
cat_features = np.where(X.dtypes != np.float64)[0]

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[valid_idx], y.iloc[valid_idx]
    
    train_pool = Pool(X_train, y_train,cat_features=cat_features)
    val_pool = Pool(X_val, y_val,cat_features=cat_features)
    
    clf = CatBoostClassifier(
    eval_metric='AUC',
    learning_rate=0.03,
    iterations=n_est)
    clf.fit(train_pool, eval_set=val_pool,verbose=300)
    
    y_pred_val = clf.predict_proba(X_val[list(X)])[:,1]
    auc_val = roc_auc_score(y_val, y_pred_val)
    print("AUC for fold ",n_fold,": ",auc_val)
    auc_vals.append(auc_val)
    
    y_pred_test = clf.predict_proba(df_test[feat_cols])[:,1]
    test_preds[n_fold, :] = y_pred_test
    print("----------------")