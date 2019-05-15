#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:15:20 2019

@author: tarun.bhavnani@dev.smecorner.com
"""

import numpy as np

#sigmoid fx
def nonlin(x,deriv=False):
  if(deriv==True):
    return x*(1-x)
  return 1/(1+np.exp(-x))

# input dataset
X = np.array([  [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
 
# output dataset           
y = np.array([[0,0,1,1]]).T
# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)
# initialize weights randomly with mean 0
w0 = 2*np.random.random((3,1)) - 1

for iter in range(10000):
  # forward propagation
  l0 = X
  l1 = nonlin(np.dot(l0,w0))
  #this is prediction
  #we do np.dot(l0,w0) which gives us a (4,1) i.e one result for all the four inputs
  #then we do a sigmoid activation, see how the results change and understand how it helps!
  
  # how much did we miss?
  l1_error = y - l1
  # multiply how much we missed by the
  #The Error Weighted Derivative
  l1_delta = l1_error * nonlin(l1,True) #is sigmoid being allpied here!!
  w0 += np.dot(l0.T,l1_delta)

print ("Output After Training:")
print (l1)



#lets create a three layer nn








