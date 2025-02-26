# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:24:16 2024

@author: tarun
"""

#conda install -c conda-forge shap
# importing the required libraries
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv(r"D:\kaggle\Big Mart AV\train.csv")

#print(df)

# imputing missing values in Item_Weight by median and Outlet_Size with mode
df['Item_Weight'].fillna(df['Item_Weight'].median(), inplace=True)
df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0], inplace=True)

# Feature Engineering
# creating a broad category of type of Items
df['Item_Type_Combined'] = df['Item_Identifier'].apply(lambda df: df[0:2])
df['Item_Type_Combined'] = df['Item_Type_Combined'].map({'FD':'Food', 'NC':'Non-Consumable', 'DR':'Drinks'})

df['Item_Type_Combined'].value_counts()

# operating years of the store
df['Outlet_Years'] = 2013 - df['Outlet_Establishment_Year']

# modifying categories of Item_Fat_Content
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF':'Low Fat', 'reg':'Regular', 'low fat':'Low Fat'})
df['Item_Fat_Content'].value_counts()

#print(df.head())


#Data Preprocessing

from sklearn.preprocessing import LabelEncoder 
# label encoding the ordinal variables
le = LabelEncoder()
df['Outlet'] = le.fit_transform(df['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])

# one hot encoding the remaining categorical variables 
df = pd.get_dummies(df, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',
                              'Item_Type_Combined','Outlet'])


#Train-Test Split

from sklearn.model_selection import train_test_split

# dropping the ID variables and variables that have been used to extract new variables
df.drop(['Item_Type','Outlet_Establishment_Year', 'Item_Identifier', 'Outlet_Identifier'],axis=1,inplace=True)

# separating the dependent and independent variables
X = df.drop('Item_Outlet_Sales',1)
y = df['Item_Outlet_Sales']

# creating the training and validation set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)








