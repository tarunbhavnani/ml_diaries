# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 15:59:54 2025

@author: tarun
"""


# =============================================================================
# SVD (Singular Value Decomposition)
# ✅ What it does:
# =============================================================================

Decomposes a matrix into three components:

A=UΣV T
 
𝑈
U = Left singular vectors (basis for row space)
Σ
Σ = Singular values (importance of each component)
𝑉
𝑇
V 
T
  = Right singular vectors (basis for column space)
✅ How it works:

Finds singular values, which tell how much information each component captures.
Works directly on the data matrix without considering variance.
Can be used for compression, noise reduction, and topic modeling.
✅ Limitations:

Purely linear—cannot capture nonlinear relationships.
Doesn’t learn new representations, just factorizes data

# =============================================================================
# PCA
# =============================================================================

 Compute the Covariance Matrix

The covariance matrix captures the relationships between the original features.
2️⃣ Find Eigenvectors and Eigenvalues

Eigenvectors: Define the new feature directions (axes of maximum variance).
Eigenvalues: Represent the importance (variance explained) of each new axis.
3️⃣ Transform Data to the New Basis

The original features are linearly combined along these new eigenvector directions.
The new features (principal components) are uncorrelated.


PCA creates new features that are linear combinations of the original features.
✅ The new features (principal components) are uncorrelated.
✅ The directions of the new features are the eigenvectors.
✅ The importance of each feature is given by its eigenvalue.


How it relates to SVD:

PCA = SVD on the covariance matrix:

𝐶=𝑉Σ2𝑉𝑇
C=VΣ 2V T
 
Principal Components = Right singular vectors (V) from SVD.

Singular values = Square root of PCA eigenvalues.
# =============================================================================
# Autoencoders (Neural Networks for Dimensionality Reduction)
# ✅ What it does:
# =============================================================================

Uses a neural network to learn a compressed representation of the data.
Unlike SVD/PCA, autoencoders do not assume linearity.
✅ How it works:
1️⃣ Encoder compresses input into a lower-dimensional latent space.
2️⃣ Decoder reconstructs the original data from this compressed form.
3️⃣ The network is trained using gradient descent to minimize reconstruction error.

✅ Key Difference from PCA/SVD:

Instead of finding fixed mathematical transformations, autoencoders learn the best transformation.
Can model complex, nonlinear structures better than PCA.
Uses backpropagation to refine representations.
✅ Limitations:

Requires training and tuning (unlike PCA/SVD, which are immediate).
Can overfit if not enough data is provided.
More computationally expensive than SVD/PCA.
