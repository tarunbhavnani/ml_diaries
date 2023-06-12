# -*- coding: utf-8 -*-
"""
Created on Mon May  8 22:08:03 2023

@author: tarun
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD

# example data matrix
X = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
], dtype=float)

# center the data matrix
X_centered = X - np.mean(X, axis=0)

# compute the truncated SVD approximation manually
U, Sigma, Vt = np.linalg.svd(X)
U.shape, Sigma.shape, Vt.shape
U[:,:2], Sigma[], Vt[:1,0]
np.dot(np.dot(U, Sigma),Vt)

# alternatively, we can use scikit-learn's TruncatedSVD class to compute the truncated SVD
svd = TruncatedSVD(n_components=k)
X_k_svd = svd.fit_transform(X)

# compare the truncated SVD approximations to the original matrix
print("Original matrix:")
print(X)
print("Truncated SVD approximation (computed manually):")
print(X_k)
print("Truncated SVD approximation (computed using scikit-learn):")
print(X_k_svd)