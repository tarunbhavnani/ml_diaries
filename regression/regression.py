#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 12:34:01 2019

@author: tarun.bhavnani
https://towardsdatascience.com/logit-of-logistic-regression-understanding-the-fundamentals-f384152a33d1
"""

import os
os.chdir('/home/tarun.bhavnani/Desktop/git_tarun/Machine_Learning')

import pandas as pd
gender_df = pd.read_csv('gender_purchase.csv')
print (gender_df.head(3))

table= pd.crosstab(gender_df['Gender'], gender_df['Purchase'])

#Odds, which describes the ratio of success to ratio of failure

purchase_Female= table.loc['Female','Yes']/(table.loc['Female','No']+table.loc['Female', 'Yes'])
no_purchase_Female= table.loc['Female','No']/(table.loc['Female','No']+table.loc['Female', 'Yes'])

Odds_Female= purchase_Female/no_purchase_Female


purchase_Male= table.loc['Male','Yes']/(table.loc['Male','No']+table.loc['Male', 'Yes'])
no_purchase_Male= table.loc['Male','No']/(table.loc['Male','No']+table.loc['Male', 'Yes'])

Odds_Male= purchase_Male/no_purchase_Male

#assert(Odds_Female== 1/Odds_Male)


#what if we take a log
from random import *
import math
random=[]
xlist = []
for i in range(1000):
 x = uniform(0,1)# choose numbers between 0 and 10 
 xlist.append(x)
 random.append(math.log(x))

import matplotlib.pyplot as plt
plt.scatter(xlist, random, c='purple',alpha=0.3,label=r'$log x$')
plt.ylabel(r'$log \, x$', fontsize=17)
plt.xlabel(r'$x$',fontsize=17)
plt.legend(fontsize=16)
plt.show()


"""We can appreciate clearly that while odds ratio can vary between 0 to positive infinity, log (odds ratio) will vary 
between [-∞, ∞]. Specifically when odds ratio lies between [0,1], log (odds ratio) is negative."""





"""Since logistic regression is about classification, i.e Y is a categorical variable. It’s clearly not possible to achieve 
such output with linear regression model (eq. 1.1), since the range on both sides do not match. Our aim is to transform the LHS 
in such a way that it matches the range of RHS, which is governed by the range of feature variables, [-∞, ∞]."""


"""For linear regression, both X and Y ranges from minus infinity to positive infinity. Y in logistic is categorical,
 or for the problem above it takes either of the two distinct values 0,1.
 First, we try to predict probability using the regression model. Instead of two distinct values now the LHS can take any values
 from 0 to 1 but still the ranges differ from the RHS."""


"""
Bring them to same scale

1) y= a+bx  #x~[-∞, ∞], y~[-∞, ∞]  #linear

2) P= a+bx #P~[0,1], x~[-∞, ∞]     #predict probabilities for logistic, 
    
3) P/(1-P)= a+bx #P~[0,∞], x~[-∞, ∞]  ##P/(1-P) can be alled O or the odds!!

4) ln(P/(1-P))= a+bx #P~[-∞, ∞], x~[-∞, ∞]
    
    P= 1/(1+ e**(-(a+bx)))  #its also the sigmoid function
    
"""

#lets see some


random1=[]
random2=[]
random3=[]
xlist = []
theta=[10, 1,0.1]
for i in range(100):
 x = uniform(-5,5)
 xlist.append(x)
 logreg1 = 1/(1+math.exp(-(theta[0]*x)))
 logreg2 = 1/(1+math.exp(-(theta[1]*x)))
 logreg3 = 1/(1+math.exp(-(theta[2]*x)))
 random1.append(logreg1)
 random2.append(logreg2)
 random3.append(logreg3)

plt.scatter(xlist, random1, marker='*',s=40, c='orange',alpha=0.5,label=r'$\theta = %3.1f$'%(theta[0]))
plt.scatter(xlist, random2, c='magenta',alpha=0.3,label=r'$\theta = %3.1f$'%(theta[1]))
plt.scatter(xlist, random3, c='navy',marker='d', alpha=0.3,label=r'$\theta = %3.1f$'%(theta[2]))
plt.axhline(y=0.5, label='P=0.5')
plt.ylabel(r'$P=\frac{1}{1+e^{-\theta \, x}}$', fontsize=19)
plt.xlabel(r'$x$',fontsize=18)
plt.legend(fontsize=16)
plt.show()





    









