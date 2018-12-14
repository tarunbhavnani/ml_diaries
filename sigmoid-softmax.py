#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 10:29:54 2018

@author: tarun.bhavnani@dev.smecorner.com
"""

def sigmoid(n):
    m=1/(1+np.exp(-n))
    return(m)

sigmoid(1)
x=[1,4,2,5,7,29]
[sigmoid(i) for i in lt ]
sum([sigmoid(i) for i in lt ])

    
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference
softmax(x)
softmax(x).sum()
def sm(x):
  ex=np.exp(x)
  return(ex/ex.sum(axis=0))
sm(x)
sm(x).sum()