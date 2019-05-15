#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:13:58 2019

@author: tarun.bhavnani@dev.smecorner.com
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
iris=load_iris()
X_train, X_test, Y_train, Y_test= train_test_split(iris.data, iris.target, test_size=.2, random_state=23)


pipe_svm = Pipeline([('scl', StandardScaler()),
			('clf', svm.SVC(random_state=42))])

pipe_svm_pca = Pipeline([('scl', StandardScaler()),
			('pca', PCA(n_components=2)),
			('clf', svm.SVC(random_state=42))])
			


param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
grid_params_svm = [{'clf__kernel': ['linear', 'rbf'], 
		'clf__C': param_range}]
jobs=-1
gs_svm = GridSearchCV(estimator=pipe_svm,
			param_grid=grid_params_svm,
			scoring='accuracy',
			cv=10,
			n_jobs=jobs, verbose=1)

gh=gs_svm.fit(X_train,Y_train)

gs_svm_pca = GridSearchCV(estimator=pipe_svm_pca,
			param_grid=grid_params_svm,
			scoring='accuracy',
			cv=10,
			n_jobs=jobs)
ghpca=gs_svm_pca.fit(X_train,Y_train)
