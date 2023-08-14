# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 13:25:59 2023

@author: tarun
"""

import pandas as pd

ipl= pd.read_csv(r"C:\Users\tarun\Desktop\PythonProjects\Dream11\IPL Ball-by-Ball 2008-2020.csv")

#takje only those matches where all 4 innings are played

ipl=ipl[ipl.inning.isin([1,2])]

valid_ids= ipl.groupby('id')['inning'].nunique().reset_index()
valid_ids=valid_ids[valid_ids.inning==2]
ipl=ipl[ipl.id.isin(list(valid_ids.id))]

# =============================================================================
# ['id', 'inning', 'over', 'ball', 'batsman', 'non_striker', 'bowler', 'batsman_runs', 'extra_runs', 
#  'total_runs', 'non_boundary', 'is_wicket', 'dismissal_kind', 'player_dismissed', 'fielder', 'extras_type', 'batting_team', 'bowling_team']
# 
# =============================================================================
hd= ipl.head(1000)

ipl['four']= [1 if i==4 else 0 for i in ipl['batsman_runs']]
ipl['six']= [1 if i==6 else 0 for i in ipl['batsman_runs']]



#runs by batsmen
# match_1.groupby("batsman")['batsman_runs'].sum()

# bat=match_1.groupby(["batsman", 'inning']).agg({"batsman_runs":sum, 'four':sum, "six":sum }).reset_index()



#bowler analysis
# match_1.groupby(["bowler", "inning"])['batsman_runs'].sum()
# bowl=match_1.groupby(["bowler", "inning"]).agg({"batsman_runs":sum, "extra_runs":sum, "is_wicket":sum, "four":sum, "six":sum, "over":"nunique"}).reset_index()


bat=ipl.groupby(["id","batsman", 'inning']).agg({"batsman_runs":sum, 'four':sum, "six":sum }).reset_index()
bat['fifty']=[1 if 50<=i<100 else 0 for i in bat.batsman_runs]
bat['century']=[1 if i>=100 else 0 for i in bat.batsman_runs]

bat=bat.groupby(['id','inning']).agg({"batsman_runs":sum, "four":sum, "six":sum, "fifty":sum, "century":sum}).reset_index()
bat.index= bat.id

bat1=bat[bat.inning==1]
bat1.drop(['id', "inning"],axis=1, inplace=True)
bat2=bat[bat.inning==2]
bat2.drop(['id', "inning"],axis=1, inplace=True)




bowl=ipl.groupby(["id","bowler", "inning"]).agg({"batsman_runs":sum, "extra_runs":sum, "is_wicket":sum, "four":sum, "six":sum, "over":"nunique"}).reset_index()
bowl_maiden=ipl.groupby(["id","bowler", "inning", "over"])["total_runs"].sum().reset_index()
bowl_maiden["maiden"]=[1 if i==0 else 0 for i in bowl_maiden.total_runs]
bowl_maiden=bowl_maiden.groupby(["id","bowler", 'inning'])['maiden'].sum().reset_index()

bowl= bowl.merge(bowl_maiden, on=["id","bowler", "inning"],  how= "left")
bowl["rr"]= [(i+j)/k for i,j,k in zip(bowl.extra_runs, bowl.batsman_runs, bowl.over)]
bowl["rr45"]= [1 if 0<=i<5 and j>=2  else 0 for i,j in zip(bowl["rr"], bowl.over)]
bowl["rr56"]= [1 if 5<=i<6 and j>=2  else 0 for i,j in zip(bowl["rr"], bowl.over)]
bowl["4wkt"]=[1 if i==4 else 0 for i in bowl.is_wicket]
bowl["5wkt"]=[1 if i>=5 else 0 for i in bowl.is_wicket]

bowl= bowl[['id', 'bowler', 'inning', 'is_wicket', 'four', 'six', 'maiden', 'rr45', 'rr56',"4wkt","5wkt"]]
bowl=bowl.groupby(['id','inning']).agg({"is_wicket":sum, "maiden":sum, "rr45":sum, "rr56":sum, "4wkt":sum,"5wkt":sum}).reset_index()
bowl.index= bowl.id


bowl1= bowl[bowl.inning==1]
bowl1.drop(['id', "inning"],axis=1, inplace=True)
bowl2= bowl[bowl.inning==2]
bowl2.drop(['id', "inning"],axis=1, inplace=True)




# [i for i in bowl1.id if i not in list(bowl2.id)]
# #[501265, 829763]


# kl=ipl[ipl.id.isin([501265, 829763])]

#points system

# points={"run":1,
#  "six":2,
#  "four":1,
#  "wicket":25,
#  "maiden":8,"rr0-5":4,"rr5-6":2, "4wkt":8, "5wkt":16, "fifty":8, "century":16, "strike50":-6, "strike60":-4, "strike70":-2}


# =============================================================================
# lets make the model
# bat_inning1+bowl_inning1==bat_inning2+bowl_inning2
# =============================================================================



#equation

sum(np.dot(bat_df_i1, bat_weights))+sum(np.dot(bowl_df_i1, bowl_weights))-sum(np.dot(bat_df_i2, bat_weights))+sum(np.dot(bowl_df_i2, bowl_weights))







# =============================================================================
# 
# =============================================================================

import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class CricketDataset(Dataset):
    def __init__(self, bat1,bat2, bowl1, bowl2):
        self.features_bat_i1 = bat1
        self.features_bat_i2 = bat2
        self.features_bowl_i1 = bowl1
        self.features_bowl_i2 = bowl2
        
        
    def __len__(self):
        return len(self.features_bat_i1)
    
    def __getitem__(self, idx):
        bat_i1 = torch.tensor(self.features_bat_i1.iloc[idx], dtype=torch.float32)
        bat_i2 = torch.tensor(self.features_bat_i2.iloc[idx], dtype=torch.float32)
        bowl_i1 = torch.tensor(self.features_bowl_i1.iloc[idx], dtype=torch.float32)
        bowl_i2 = torch.tensor(self.features_bowl_i2.iloc[idx], dtype=torch.float32)
        
        
        return bat_i1, bat_i2,bowl_i1,bowl_i2

# Create datasets
dataset_ = CricketDataset(bat1,bat2, bowl1, bowl2)


# Create data loaders
batch_size = 32  # You can adjust this based on your needs
dataloader = DataLoader(dataset_, batch_size=batch_size, shuffle=True)




# Define the objective function that you want to minimize
def objective_function(bat_features_i1, bat_features_i2, bowl_features_i1, bowl_features_i2, bat_weights, bowl_weights):
    
    bat_diff = torch.sum(torch.matmul(bat_features_i1, bat_weights)) + torch.sum(torch.matmul(bat_features_i2, bat_weights))
    bowl_diff = torch.sum(torch.matmul(bowl_features_i1, bowl_weights)) + torch.sum(torch.matmul(bowl_features_i2, bowl_weights))
    return bat_diff - bowl_diff

def objective_function(bat_features_i1, bat_features_i2, bowl_features_i1, bowl_features_i2, bat_weights, bowl_weights):
    bat_diff = torch.mean(torch.matmul(bat_features_i1, bat_weights)) - torch.mean(torch.matmul(bat_features_i2, bat_weights))
    bowl_diff = torch.mean(-torch.matmul(bowl_features_i1, bowl_weights)) + torch.mean(torch.matmul(bowl_features_i2, bowl_weights))
    return bat_diff + bowl_diff


class WeightOptimizationModel(nn.Module):
    def __init__(self, input_dim_bat, input_dim_bowl):
        super(WeightOptimizationModel, self).__init__()
        self.bat_weight = torch.tensor([1,1,2,8,16],dtype=torch.float32)
        self.bowl_weight = nn.Parameter(torch.randn(6,1))

    def forward(self, bat_features_i1, bat_features_i2, bowl_features_i1, bowl_features_i2):
        return objective_function(bat_features_i1, bat_features_i2, bowl_features_i1, bowl_features_i2,
                                  self.bat_weight, self.bowl_weight)


# ... Instantiate the model, optimizer, and other setup as before ...

input_dim_bat = bat1.shape[1]
input_dim_bowl = bowl1.shape[1]
model = WeightOptimizationModel(input_dim_bat, input_dim_bowl)
optimizer = optim.SGD(model.parameters(), lr=0.005)



# Training loop
epochs = 1000
#epoch=1
for epoch in range(epochs):
    model.train()
    total_loss = []

    for bat_i1, bat_i2,bowl_i1,bowl_i2 in dataloader:
        optimizer.zero_grad()
        loss = model(bat_i1, bat_i2, bowl_i1, bowl_i2)
        loss.backward()
        optimizer.step()
        
        total_loss.append(loss.item())
        print(loss.item())
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Extract the learned weights
learned_bat_weights = model.bat_weight.detach().numpy()
learned_bowl_weights = model.bowl_weight.detach().numpy()

print("Learned Batsman Weights:", learned_bat_weights)
print("Learned Bowler Weights:", learned_bowl_weights)


# =============================================================================
# 
# =============================================================================

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CricketDataset(Dataset):
    def __init__(self, bat1,bat2, bowl1, bowl2):
        self.features_bat_i1 = bat1
        self.features_bat_i2 = bat2
        self.features_bowl_i1 = bowl1
        self.features_bowl_i2 = bowl2
        
        
    def __len__(self):
        return len(self.features_bat_i1)
    
    def __getitem__(self, idx):
        bat_i1 = torch.tensor(self.features_bat_i1.iloc[idx], dtype=torch.float32)
        bat_i2 = torch.tensor(self.features_bat_i2.iloc[idx], dtype=torch.float32)
        bowl_i1 = torch.tensor(self.features_bowl_i1.iloc[idx], dtype=torch.float32)
        bowl_i2 = torch.tensor(self.features_bowl_i2.iloc[idx], dtype=torch.float32)
        
        
        return bat_i1, bat_i2,bowl_i1,bowl_i2

# Create datasets
dataset_ = CricketDataset(bat1,bat2, bowl1, bowl2)


# Create data loaders
batch_size = 32  # You can adjust this based on your needs
dataloader = DataLoader(dataset_, batch_size=batch_size, shuffle=True)


# Define the objective function with L2 regularization
def objective_function(bat_features_i1, bat_features_i2, bowl_features_i1, bowl_features_i2, bat_weights, bowl_weights, l2_lambda):
    bat_diff = torch.sum(torch.matmul(bat_features_i1, bat_weights)) - torch.sum(torch.matmul(bat_features_i2, bat_weights))
    bowl_diff = torch.sum(torch.matmul(bowl_features_i1, bowl_weights)) - torch.sum(torch.matmul(bowl_features_i2, bowl_weights))
    l2_reg = l2_lambda * (torch.norm(bat_weights) + torch.norm(bowl_weights))  # L2 regularization term
    return bat_diff + bowl_diff + l2_reg

class WeightOptimizationModel(nn.Module):
    def __init__(self, input_dim_bat, input_dim_bowl):
        super(WeightOptimizationModel, self).__init__()
        self.bat_weight = nn.Parameter(torch.randn(input_dim_bat, 1)).to(device)
        self.bowl_weight = nn.Parameter(torch.randn(input_dim_bowl, 1)).to(device)

    def forward(self, bat_features_i1, bat_features_i2, bowl_features_i1, bowl_features_i2, l2_lambda):
        return objective_function(bat_features_i1, bat_features_i2, bowl_features_i1, bowl_features_i2,
                                  self.bat_weight, self.bowl_weight, l2_lambda)

# ... Instantiate the model, optimizer, and other setup as before ...
model = WeightOptimizationModel(input_dim_bat, input_dim_bowl)
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Regularization strength (lambda)
l2_lambda = 0.01

# Training loop
epochs = 1000
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for bat_i1, bat_i2,bowl_i1,bowl_i2 in dataloader:
        bat_i1, bat_i2 = bat_i1.to(device), bat_i2.to(device)
        bowl_i1, bowl_i2 = bowl_i1.to(device), bowl_i2.to(device)

        optimizer.zero_grad()
        loss = model(bat_i1, bat_i2, bowl_i1, bowl_i2, l2_lambda)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")

# Extract the learned weights
learned_bat_weights = model.bat_weight.cpu().detach().numpy()
learned_bowl_weights = model.bowl_weight.cpu().detach().numpy()

print("Learned Batsman Weights:", learned_bat_weights)
print("Learned Bowler Weights:", learned_bowl_weights)
