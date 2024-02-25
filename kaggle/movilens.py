# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 17:38:06 2024

@author: tarun
"""

import os
import pandas as pd


ratings= pd.read_csv(r"D:\kaggle\movilens\ratings.csv")
movies= pd.read_csv(r"D:\kaggle\movilens\movies.csv")


interactions_matrix_size= ratings['userId'].nunique()*ratings["movieId"].nunique()
interactions_count= ratings.shape[0]
sparsity=1-(interactions_count/interactions_matrix_size)



top_users= ratings.groupby("userId")['rating'].count().sort_values(ascending=False)[:10]
top_movies= ratings.groupby("movieId")['rating'].count().sort_values(ascending=False)[:10]



jk=ratings.groupby(['userId','movieId'])['rating'].sum().reset_index()
#jk=ratings.groupby(['userId', 'movieId'])['rating'].sum().unstack()

user_count= len(set(jk.userId))
movie_count= len(set(jk.movieId))
# Pivot the DataFrame
pivot_df = jk.pivot(index='userId', columns='movieId', values='rating')








ratings_expanded= pd.merge(ratings, movies, on="movieId")
ratings_stats= pd.DataFrame(ratings_expanded.groupby("title")['rating'].mean())
ratings_stats['rating_count']=pd.DataFrame(ratings_expanded.groupby("title")['rating'].count())

import seaborn as sns

sns.displot(data=ratings_stats, x= "rating_count", bins=50, height=6, aspect=2)


sns.displot(data= ratings_stats, x="rating_count", log_scale=True, height=6, aspect=2)


sns.displot(data= ratings_stats, x="rating" ,height=6, aspect=2)


sns.jointplot(data= ratings_stats, x='rating', y='rating_count', alpha=.5, height=8 )


# =============================================================================
# Popularity 
# =============================================================================


#rating for a user will be the average rating of that movie\
    
Popularity=ratings_expanded.groupby('movieId')['rating'].mean().reset_index()


#evaluate(rmse) by split here...



# =============================================================================
# NCF, neural collaboratibe filtering
# =============================================================================

# input layer takes user and item ids
# embedding layer projects sparse into dense vectors
# dense user and item vetors are latent reprsentations 
# neural cf layers are mlp that model user-item interactions
# output layer is outpur score

import numpy as np


X=ratings[['userId', 'movieId']]
Y= ratings['rating'].astype(np.float32)



from sklearn.model_selection import train_test_split
random_state=7
test_size=.2


X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size= test_size, random_state=random_state)
datasets= {'train':(X_train, Y_train), 'test':(X_test, Y_test)}



#model autoring
import torch
from torch import nn

class NeuralColabFiltering(nn.Module):
    
    
    def __init__(self, user_count, movie_count, embedding_size=32,
                 hidden_layers=(64,32,16,8),
                 dropout_rate=None,
                 output_range=(1,5)):
        super().__init__()
        
        #initialize embedding hash size
        self.user_hash_size=user_count
        self.movie_hash_size=movie_count
        
        #initialize model architecture
        self.user_embedding= nn.Embedding(user_count, embedding_size)
        self.movie_embedding= nn.Embedding(movie_count, embedding_size)
        self.MLP= self._gen_MLP(embedding_size, hidden_layers, dropout_rate)
        if (dropout_rate):
            self.dropout=nn.Dropout(dropout_rate)
        
        #initialize output normalization parameters
        
        assert output_range and len(output_range)==2
        self.norm_min=min(output_range)
        self.norm_range=abs(output_range[0]-output_range[1])+1
        
        self._init_params()
        
        
    def _gen_MLP(self, embedding_size, hidden_layers_units, dropout_rate):
        
        assert (embedding_size*2)==hidden_layers_units[0]
        
        hidden_layers=[]
        input_units=hidden_layers_units[0]
        
        for num_units in hidden_layers_units[1:]:
            hidden_layers.append(nn.Linear(input_units, num_units))
            hidden_layers.append(nn.ReLU())
            if (dropout_rate):
                hidden_layers.append(nn.Droup(dropout_rate))
            input_units= num_units
            
        hidden_layers.append(nn.Linear(hidden_layers_units[-1],1))
        hidden_layers.append(nn.Sigmoid())
        
        return nn.Sequential(*hidden_layers)
        
    def _init_params(self):
        def weights_init(m):
            if type(m)==nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            
        
        self.user_embedding.weight.data.uniform_(-0.05, 0.05)
        self.movie_embedding.weight.data.uniform_(-0.05, 0.05)
        self.MLP.apply(weights_init)
    
    
    def forward(self, user_id, movie_id):
        user_feature= self.user_embedding(user_id % self.user_hash_size)#dense features
        movie_feature= self.movie_embedding(movie_id % self.movie_hash_size)#dense features
        
        x= torch.cat([user_feature, movie_feature], dim=1)
        if hasattr(self, 'dropout'):
            x= self.dropout(x)
        x= self.MLP(x)
        normalized_output= x*self.norm_range+self.norm_min
        
        return normalized_output
            
                    
        
        
ncf= NeuralColabFiltering(user_count, movie_count)
            
        
        
        
num_params= sum(p.numel() for p in ncf.parameters())

print(f'Number of params:{num_params:,}, model training size :{num_params*4/(1024**2):.2f}MB')


from random import randrange

ncf.eval()


ratings_row= randrange(0, ratings.shape[0]-1)
test_user= int(ratings.iloc[ratings_row].userId)
test_movie= int(ratings.iloc[ratings_row].movieId)
actual_rating= ratings.iloc[ratings_row].rating

ncf.to('cpu')

predicted_rating= ncf(torch.tensor([test_user]), torch.tensor([test_movie]))













