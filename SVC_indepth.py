#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:29:54 2019

@author: tarun.bhavnani@dev.smecorner.com
https://medium.com/all-things-ai/in-depth-parameter-tuning-for-svc-758215394769
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
%matplotlib inline
# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2] # we only take the first two features. We could
 # avoid this ugly slicing by using a two-dim dataset
y = iris.target

def plotSVC(title):
  # create a mesh to plot in
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  h = (x_max / x_min)/100
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
  np.arange(y_min, y_max, h))
  plt.subplot(1, 1, 1)
  Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)
  plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
  plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
  plt.xlabel('Sepal length')
  plt.ylabel('Sepal width')
  plt.xlim(xx.min(), xx.max())
  plt.title(title)
  plt.show()
  
#kernels  
kernels = ['linear', 'rbf', 'poly']
for kernel in kernels:
  svc = svm.SVC(kernel=kernel).fit(X, y)
  plotSVC('kernel=' + str(kernel))


#Gamma
  
gammas = [0.1, 1, 10, 100]
for gamma in gammas:
   svc = svm.SVC(kernel='rbf', gamma=gamma).fit(X, y)
   plotSVC('gamma=' + str(gamma))
   
#C
   
cs = [0.1, 1, 10, 100, 1000]
for c in cs:
   svc = svm.SVC(kernel='rbf', C=c).fit(X, y)
   plotSVC('C=' + str(c))

#degree

degrees = [0, 1, 2, 3, 4, 5, 6]
for degree in degrees:
   svc = svm.SVC(kernel='poly', degree=degree).fit(X, y)
   plotSVC('degree=' + str(degree))


print("Fitting the classifier to the training set")
t0 = time()
#param_grid = {'kernel' : ['linear', 'rbf', 'poly'],'gamma' : [0.1, 1, 10, 100],'C' : [0.1, 1, 10, 100,1000],'degree' : [0, 1, 2, 3, 4, 5, 6] }
#with kernel in param_grid, it took a lot longer.
param_grid = {'gamma' : [0.1, 1, 10, 100],'C' : [0.1, 1, 10, 100,1000],'degree' : [0, 1, 2, 3, 4, 5, 6] }
clf = GridSearchCV(SVC(class_weight='balanced'),
                   param_grid, cv=5)
clf = clf.fit(X, y)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)



