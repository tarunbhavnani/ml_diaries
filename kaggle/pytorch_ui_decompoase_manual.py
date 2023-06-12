# -*- coding: utf-8 -*-
"""
Created on Mon May  8 12:07:56 2023

@author: tarun
"""
import numpy as np
user_item_matrix = np.array([
    [3, 0, 2, 0, 1],
    [0, 2, 3, 1, 0],
    [2, 0, 1, 0, 3],
    [0, 1, 0, 3, 2]
], dtype=float)


#5*4
# =============================================================================
# A = U . Sigma . V^T
# =============================================================================


#k=2
#5*3, 2*2, 2*4


u= np.random.rand(5,2)
sigma= np.diag(np.random.rand(2))
vt= np.random.rand(2,4)

a= np.dot(u,np.dot(sigma, vt))


# =============================================================================
# 
# =============================================================================

#convert ui matrix to pytorch tensor

import torch

ui=torch.from_numpy(user_item_matrix)


#set rank for matrix and iunitial;ize u, sigma and vt


rank=2

u=torch.randn(ui.shape[0], rank, requires_grad=True)
sigma= torch.randn(rank, rank, requires_grad=True)
vt= torch.randn(rank, ui.shape[1], requires_grad=True)


#set optimization and learning function

learning_rate=.01
optimizer= torch.optim.Adam([u, sigma, vt], lr= learning_rate)
# if give torch.optim.Adam([U, V], lr=learning_rate) instead of torch.optim.Adam([U, Sigma, V], lr=learning_rate)
# does that mean that the model will only update values of U and V and not sigma
# Yes, the optimizer will only update the values of U and V, and not Sigma. This is because Sigma is not included in the 
# list of parameters passed to the optimizer.

#define loss function


def loss_(org, reconst):
  return torch.mean((org-reconst)**2)
  


#train mdoel


for epoch in range(100):
    reconstructed= torch.mm(torch.mm(u,sigma), vt)
    loss= loss_(ui, reconstructed)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1)%10==0:
        print("Epoch: {}/{},Loss: {:.4f}".format(epoch+1,100, loss.item()))
    
u=u.detach().numpy()
sigma=sigma.detach().numpy()
vt=vt.detach().numpy()

# =============================================================================
# lets try diff optimizers
# =============================================================================


import torch
import numpy as np

user_item_matrix = np.array([
    [3, 0, 2, 0, 1],
    [0, 2, 3, 1, 0],
    [2, 0, 1, 0, 3],
    [0, 1, 0, 3, 2]
], dtype=float)

user_item_tensor = torch.from_numpy(user_item_matrix)

rank = 4

learning_rate = 0.01
num_epochs = 1000

optimizers = [
    torch.optim.Adam,
    torch.optim.RMSprop,
    torch.optim.SGD,
    torch.optim.Adagrad
]

for optimizer_class in optimizers:
    print(f'Training with optimizer: {optimizer_class.__name__}')
    U = torch.randn(user_item_tensor.shape[0], rank, requires_grad=True)
    Sigma = torch.randn(rank, rank, requires_grad=True)
    V = torch.randn(user_item_tensor.shape[1], rank, requires_grad=True)

    optimizer = optimizer_class([U, Sigma, V], lr=learning_rate)
    for epoch in range(num_epochs):
        reconstructed = torch.mm(torch.mm(U, Sigma), V.t())
        loss = torch.mean((user_item_tensor - reconstructed)**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

    print('Optimization finished.')
    U=U.detach().numpy()
    Sigma=Sigma.detach().numpy()
    V=V.detach().numpy()
    
    print(np.dot(np.dot(U, Sigma), V.T))
