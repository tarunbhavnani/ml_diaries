# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 13:27:27 2022

@author: ELECTROBOT
"""


# load dataset
import os
os.chdir(r'C:\Users\ELECTROBOT\Desktop\kaggle\Song_popularity')

train = pd.read_csv('train.csv', index_col=0)
#drop na rows for now
train=train[~train.apply(lambda x: sum(x.isnull())>0, axis=1)]

y= train['song_popularity']
X= train.drop(['song_popularity'], axis=1)

col_cat = ['key', 'audio_mode', 'time_signature']

col_num= [i for i in train if i not in col_cat]



from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

steps = [('scaler', StandardScaler()), ('SVM', SVC())]

pipeline = Pipeline(steps) # define the pipeline object.

Y= train['song_popularity']
X= train.drop(['song_popularity'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=30, stratify=Y)


parameteres = {'SVM__C':[0.001,0.1,10], 'SVM__gamma':[0.1,0.01]}


grid = GridSearchCV(pipeline, param_grid=parameteres, cv=3, verbose=5)
grid.fit(X_train, y_train)

print ("score = %3.2f" %(grid.score(X_test,y_test)))
print (grid.best_params_)
