# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 11:48:44 2024

@author: tarun
"""
import numpy as np
import pandas as pd
from collections import Counter

# Load the Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target

# Define a class for a node in the decision tree
class TreeNode:
    def __init__(self, feature_index=None, threshold=None, value=None, left=None, right=None):
        self.feature_index = feature_index  # Index of the feature to split on
        self.threshold = threshold          # Threshold value for the split
        self.value = value                  # Class label if node is a leaf
        self.left = left                    # Left child (subtree)
        self.right = right                  # Right child (subtree)

# Define a function to calculate Gini impurity
def gini_impurity(tr):
    classes = np.unique(tr)
    n = len(tr)
    impurity = 1
    for c in classes:
        p = (tr == c).sum() / n
        
        impurity -= p**2
        print(c,p,impurity)
    return impurity

# Define a function to find the best split, bases on least gini impurity
def find_best_split(X, y):
    best_gini = float('inf')
    best_feature_index = None
    best_threshold = None
    
    #goes through each feature one by one
    
    for feature_index in range(X.shape[1]):
        #break
        
        thresholds = np.unique(X[:, feature_index])
        
        #goes through each unique value in feature one by one
        for threshold in thresholds:
            #break
            
            #in each unique value for a feature gets all the labels on left and right and calculates the gini impurity
            #saves the one split with the lest gini impurity
            
            left_indices = np.where(X[:, feature_index] <= threshold)[0]
            right_indices = np.where(X[:, feature_index] > threshold)[0]

            gini_left = gini_impurity(y[left_indices])
            gini_right = gini_impurity(y[right_indices])

            gini = len(left_indices) / len(y) * gini_left + len(right_indices) / len(y) * gini_right

            if gini < best_gini:
                best_gini = gini
                best_feature_index = feature_index
                best_threshold = threshold

    return best_feature_index, best_threshold

# Define a function to build the decision tree
def build_tree(X, y):
    if len(np.unique(y)) == 1:
        return TreeNode(value=y[0])

    if len(X) == 0:
        return TreeNode(value=Counter(y).most_common(1)[0][0])

    best_feature_index, best_threshold = find_best_split(X, y)

    if best_feature_index is None:
        return TreeNode(value=Counter(y).most_common(1)[0][0])

    left_indices = np.where(X[:, best_feature_index] <= best_threshold)[0]
    right_indices = np.where(X[:, best_feature_index] > best_threshold)[0]

    left_subtree = build_tree(X[left_indices], y[left_indices])
    right_subtree = build_tree(X[right_indices], y[right_indices])

    return TreeNode(feature_index=best_feature_index, threshold=best_threshold, left=left_subtree, right=right_subtree)

# Define a function to make predictions with the decision tree
def predict_tree(tree, x):
    if tree.value is not None:
        return tree.value
    
    if x[tree.feature_index] <= tree.threshold:
        return predict_tree(tree.left, x)
    else:
        return predict_tree(tree.right, x)

# Build the decision tree
decision_tree = build_tree(X, y)

# Make predictions
predictions = [predict_tree(decision_tree, x) for x in X]

# Calculate accuracy
accuracy = np.mean(predictions == y)
print("Accuracy:", accuracy)


# =============================================================================
# regression 
# =============================================================================


import numpy as np

# Define a class for a node in the decision tree
class TreeNode:
    def __init__(self, feature_index=None, threshold=None, value=None, left=None, right=None):
        self.feature_index = feature_index  # Index of the feature to split on
        self.threshold = threshold          # Threshold value for the split
        self.value = value                  # Predicted value if node is a leaf
        self.left = left                    # Left child (subtree)
        self.right = right                  # Right child (subtree)

# Define a function to calculate mean squared error (MSE)
def mean_squared_error(y):
    return np.mean((y - np.mean(y))**2)

# Define a function to find the best split
def find_best_split(X, y):
    best_mse = float('inf')
    best_feature_index = None
    best_threshold = None

    for feature_index in range(X.shape[1]):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            left_indices = np.where(X[:, feature_index] <= threshold)[0]
            right_indices = np.where(X[:, feature_index] > threshold)[0]

            mse_left = mean_squared_error(y[left_indices])
            mse_right = mean_squared_error(y[right_indices])

            mse = len(left_indices) / len(y) * mse_left + len(right_indices) / len(y) * mse_right

            if mse < best_mse:
                best_mse = mse
                best_feature_index = feature_index
                best_threshold = threshold

    return best_feature_index, best_threshold

# Define a function to build the decision tree
def build_tree(X, y):
    if mean_squared_error(y) < 1e-5:  # Stopping criterion (minimum MSE)
        return TreeNode(value=np.mean(y))

    best_feature_index, best_threshold = find_best_split(X, y)

    if best_feature_index is None:
        return TreeNode(value=np.mean(y))

    left_indices = np.where(X[:, best_feature_index] <= best_threshold)[0]
    right_indices = np.where(X[:, best_feature_index] > best_threshold)[0]

    left_subtree = build_tree(X[left_indices], y[left_indices])
    right_subtree = build_tree(X[right_indices], y[right_indices])

    return TreeNode(feature_index=best_feature_index, threshold=best_threshold, left=left_subtree, right=right_subtree)

# Define a function to make predictions with the decision tree
def predict_tree(tree, x):
    if tree.value is not None:
        return tree.value
    
    if x[tree.feature_index] <= tree.threshold:
        return predict_tree(tree.left, x)
    else:
        return predict_tree(tree.right, x)

# Generate example data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X.squeeze() + np.random.randn(100)  # Linear relationship with noise

# Build the decision tree
decision_tree = build_tree(X, y)

# Make predictions
predictions = [predict_tree(decision_tree, x) for x in X]

# Calculate mean squared error
mse = np.mean((predictions - y) ** 2)
print("Mean Squared Error:", mse)
