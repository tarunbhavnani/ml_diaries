# -*- coding: utf-8 -*-
"""
Created on Sun May  7 16:57:28 2023

@author: tarun
"""

import numpy as np
from scipy.sparse.linalg import svds

# Example user-item matrix
user_item_matrix = np.array([
    [3, 0, 2, 0, 1],
    [0, 2, 3, 1, 0],
    [2, 0, 1, 0, 3],
    [0, 1, 0, 3, 2]
], dtype=float)

# Define number of latent factors
k = 2

# Perform matrix factorization using ALS
u, s, vt = svds(user_item_matrix, k)

# Convert s from a 1D array to a diagonal matrix
s_diag = np.diag(s)

# Compute U and V matrices
u_matrix1 = np.dot(u, s_diag)
v_matrix1 = vt.T

# Print U and V matrices
print(u_matrix1)
print(v_matrix1)
np.dot(u_matrix1, v_matrix1.T)


#manualyy

import numpy as np

# Example user-item matrix
user_item_matrix = np.array([
    [3, 0, 2, 0, 1],
    [0, 2, 3, 1, 0],
    [2, 0, 1, 0, 3],
    [0, 1, 0, 3, 2]
])

# Define number of latent factors
k = 2

# Define learning rate
alpha = 0.01

# Define regularization parameter
lambda_ = 0.1

# Initialize U and V matrices with random values
n_users, n_items = user_item_matrix.shape
u_matrix = np.random.rand(n_users, k)
v_matrix = np.random.rand(n_items, k)

# Perform matrix factorization using stochastic gradient descent
for epoch in range(1000):
    for i in range(n_users):
        for j in range(n_items):
            if user_item_matrix[i, j] != 0:
                error = user_item_matrix[i, j] - np.dot(u_matrix[i], v_matrix[j])
                u_matrix[i] += alpha * (error * v_matrix[j] - lambda_ * u_matrix[i])
                v_matrix[j] += alpha * (error * u_matrix[i] - lambda_ * v_matrix[j])

# Print U and V matrices
print(u_matrix)
print(v_matrix)

np.dot(u_matrix, v_matrix.T)
