#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:49:22 2019

@author: tarun.bhavnani
"""
#https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e
import os
os.chdir('/home/tarun.bhavnani/Desktop/git_tarun/regression')
import numpy as np

# Data Generation
np.random.seed(42)
x = np.random.rand(100, 1)
y = 1 + 2 * x + .1 * np.random.randn(100, 1)

# Shuffles the indices
idx = np.arange(100)
np.random.shuffle(idx)

# Uses first 80 random indices for train
train_idx = idx[:80]
# Uses the remaining indices for validation
val_idx = idx[80:]

# Generates train and validation sets
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]
# Initializes parameters "a" and "b" randomly
np.random.seed(42)
a = np.random.randn(1)
b = np.random.randn(1)

print(a, b)

# Sets learning rate
lr = 1e-1
# Defines number of epochs
n_epochs = 1000

for epoch in range(n_epochs):
    # Computes our model's predicted output
    yhat = a + b * x_train
    
    # How wrong is our model? That's the error! 
    error = (y_train - yhat)
    # It is a regression, so it computes mean squared error (MSE)
    loss = (error ** 2).mean()
    
    # Computes gradients for both "a" and "b" parameters
    a_grad = -2 * error.mean()
    b_grad = -2 * (x_train * error).mean()
    
    # Updates parameters using gradients and the learning rate
    a = a - lr * a_grad
    b = b - lr * b_grad
    
    print(a, b)

# Sanity Check: do we get the same results as our gradient descent?
from sklearn.linear_model import LinearRegression
linr = LinearRegression()
linr.fit(x_train, y_train)
print(linr.intercept_, linr.coef_[0])


# =============================================================================
# use matrix multiplication for the same 
# =============================================================================
# Add a column of ones to x_train to account for the intercept term
X = np.hstack((np.ones((x_train.shape[0], 1)), x_train))  # Shape (n, 2)

# Initialize parameters (a and b) as a single vector
theta = np.zeros((2, 1))  # Shape (2,1) for [a, b]

# np.random.seed(42)
# a = np.random.randn(1)
# b = np.random.randn(1)
# theta=np.asarray([a,b])
# Hyperparameters
n_epochs = 1000
lr = 0.1

for epoch in range(n_epochs):
    # Compute predictions using matrix multiplication
    yhat = X @ theta  # Shape (n,1) = (n,2) @ (2,1)

    # Compute error
    error = y_train - yhat

    # Compute gradient using matrix multiplication
    gradient = (-2 / len(x_train)) * (X.T @ error)  # Shape (2,1) = (2,n) @ (n,1)

    # Update parameters
    theta = theta - lr * gradient

    # Print updated parameters
    if epoch % 100 == 0:  # Print every 100 epochs
        print(f"Epoch {epoch}: a = {theta[0][0]}, b = {theta[1][0]}")
