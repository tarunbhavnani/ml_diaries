# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 10:29:59 2023
https://www.ethanrosenthal.com/2017/06/20/matrix-factorization-in-pytorch/

@author: ELECTROBOT
"""

import numpy as np
from scipy.sparse import rand as sprand
import torch

# Make up some random explicit feedback ratings
# and convert to a numpy array
n_users = 1_000
n_items = 1_000
ratings = sprand(n_users, n_items, density=0.01, format="csr")
ratings.data = np.random.randint(1, 5, size=ratings.nnz).astype(np.float64)
ratings = ratings.toarray()


class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, n_factors, sparse=True)
        self.item_factors = torch.nn.Embedding(n_items, n_factors, sparse=True)

    def forward(self, user, item):
        return (self.user_factors(user) * self.item_factors(item)).sum(1)

#(user_factors(user) * item_factors(item)).sum(1)
# hj=user_factors(user) * item_factors(item)
# hj=hj.detach().numpy()
# sum([i for i in hj[0]])

model = MatrixFactorization(n_users, n_items, n_factors=20)

loss_func = torch.nn.MSELoss()


optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)  # learning rate


# Sort our data
rows, cols = ratings.nonzero()
p = np.random.permutation(len(rows))
rows, cols = rows[p], cols[p]

for row, col in zip(*(rows, cols)):
    print(row, col)
   
    # Set gradients to zero
    optimizer.zero_grad()
    
    # Turn data into tensors
    rating = torch.FloatTensor([ratings[row, col]])
    row = torch.LongTensor([row])
    col = torch.LongTensor([col])

    # Predict and calculate loss
    prediction = model(row, col)
    loss = loss_func(prediction, rating)

    # Backpropagate
    loss.backward()

    # Update the parameters
    optimizer.step()


#recreate matrix

matrix_r=np.zeros((ratings.shape[0],ratings.shape[1]))
for row in range(ratings.shape[0]):
    for col in range(ratings.shape[1]):
        print(row,col)
        row = torch.LongTensor([row])
        col = torch.LongTensor([col])
        with torch.no_grad():
            outputs= model(row, col)
        matrix_r[row,col]=outputs.detach().numpy()
            
        
            



class BiasedMatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, n_factors, sparse=True)
        self.item_factors = torch.nn.Embedding(n_items, n_factors, sparse=True)
        self.user_biases = torch.nn.Embedding(n_users, 1, sparse=True)
        self.item_biases = torch.nn.Embedding(n_items, 1, sparse=True)

    def forward(self, user, item):
        pred = self.user_biases(user) + self.item_biases(item)
        pred += (
            (self.user_factors(user) * self.item_factors(item))
            .sum(dim=1, keepdim=True)
        )
        return pred.squeeze()

