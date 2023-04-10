# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:53:51 2023

@author: ELECTROBOT
"""

import torch
import torch.nn as nn
import torch.optim as optim

# define hyperparameters
n_user = 500
n_movie = 1000
embedding_size = 32
reg = 0.01
lr = 0.001
epochs = 30
batch_size=20

# define model
class Recommender(nn.Module):
    def __init__(self, n_user, n_movie, embedding_size):
        super(Recommender, self).__init__()
        self.user_embedding = nn.Embedding(n_user, embedding_size)
        self.movie_embedding = nn.Embedding(n_movie, embedding_size)
        self.user_bias_embedding = nn.Embedding(n_user, 1)
        self.movie_bias_embedding = nn.Embedding(n_movie, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, user, movie):
        u = self.user_embedding(user)
        m = self.movie_embedding(movie)
        u_bias = self.user_bias_embedding(user)
        m_bias = self.movie_bias_embedding(movie)
        dot = torch.sum(torch.mul(u, m), dim=1)
        return dot + u_bias.squeeze() + m_bias.squeeze() + self.global_bias

# initialize model
model = Recommender(n_user, n_movie, embedding_size)

# define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg)

# generate dummy data
X = torch.randint(0, n_user, (10000,))
Y = torch.randint(0, n_movie, (10000,))
R = torch.randint(0, 5, (10000,)).float()


# train model for 5 epochs
for epoch in range(epochs):
    running_loss = 0.0
    #for i in range(len(X)):
    for start in range(0, X.shape[0] - batch_size, batch_size):

        end = start + batch_size
        optimizer.zero_grad()
        output = model(X[start:end], Y[start:end])
        loss = criterion(output, R[start:end])#meansqerror, but takes zeros as well
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('Epoch %d loss: %.3f' % (epoch+1, running_loss/len(X)))
    

# =============================================================================
# 
# =============================================================================















