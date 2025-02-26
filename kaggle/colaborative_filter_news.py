# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 18:15:09 2023

@author: tarun
"""

import pandas as pd
import numpy as np
import scipy  
import pandas as pd
import math
import random
import sklearn
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt



articles_df = pd.read_csv(r"D:\kaggle\Deskdrop\shared_articles.csv")
articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED']
articles_df.head(5)
list(articles_df)
interactions_df = pd.read_csv(r"D:\kaggle\Deskdrop\users_interactions.csv")
interactions_df.head(10)
list(interactions_df)

# sum(interactions_df.contentId.isin(articles_df.contentId))/len(interactions_df)
# sum(articles_df.contentId.isin(interactions_df.contentId))/len(articles_df)

#assign severity
event_type_strength = {
   'VIEW': 1.0,
   'LIKE': 2.0, 
   'BOOKMARK': 2.5, 
   'FOLLOW': 3.0,
   'COMMENT CREATED': 4.0,  
}


interactions_df['eventStrength'] = interactions_df["eventType"].apply(lambda x: event_type_strength[x])

kl=interactions_df.groupby(['personId', 'contentId'])['eventStrength'].apply(lambda x: sum(x)).reset_index()
jk=interactions_df.groupby(['contentId'])['eventStrength'].apply(lambda x: sum(x)).reset_index()

list(interactions_df)

#collaborative filtering

#1) user item matrix

#mat= interactions_df[["personId","contentId","eventStrength"  ]]

mat=pd.pivot_table(data=interactions_df, values="eventStrength", index="personId",columns="contentId")
mat.fillna(0, inplace=True)


user= mat.loc[-1479311724257856983]
user=np.array(user).reshape(1,-1)

scores=cosine_similarity(user, mat)
scores=[(num,i) for num,i in enumerate(scores[0])]

scores=sorted(scores, key=lambda x: x[1])[::-1]
top_100=[i[0] for i in scores[:100]]

similar_users=interactions_df.iloc[top_100]

cont=similar_users.groupby("contentId").agg({"eventStrength":"sum"}). reset_index()
cont= cont.sort_values(by="eventStrength", ascending=False)[0:1000]
cont= cont.merge(articles_df, left_on="contentId", right_on="contentId", how="left")
cont.text

#eg: user like you read this as well!!

#2) item item matrix

ui=pd.pivot_table(data=interactions_df, values="eventStrength", index="personId",columns="contentId")
ui.fillna(0, inplace=True)

ii=pd.DataFrame(cosine_similarity(ui.T,ui.T))

ii.index= list(ui)
ii.columns= list(ui)
#lets ee user -1479311724257856983
user= mat.loc[-1479311724257856983]
cont_=user[user==max(user)].index[0]

cont=ii.loc[cont_]
cont=pd.DataFrame(cont).reset_index()
cont.columns= ["contentId", "score"]
cont= cont.sort_values(by="score", ascending=False)[0:1000]
cont= cont.merge(articles_df, left_on="contentId", right_on="contentId", how="left")
cont.text


#3) Matrix Factorization

ui=pd.pivot_table(data=interactions_df, values="eventStrength", index="personId",columns="contentId")
ui.fillna(0, inplace=True)

matrix= ui.values
u, s, v = svds(matrix)
u.shape, v.shape, s.shape

matrix_reconstructed= np.dot(np.dot(u, np.diag(s)), v)
#normalize
matrix_reconstructed_norm = (matrix_reconstructed - matrix_reconstructed.min()) / (matrix_reconstructed.max() - matrix_reconstructed.min())

ui_reconstructed=pd.DataFrame(matrix_reconstructed_norm)
ui_reconstructed=ui_reconstructed.set_index(ui.index)
ui_reconstructed.columns= ui.columns


cont=ui_reconstructed.loc[-1479311724257856983].reset_index()
cont.columns=["contentId", "score"]

related_content=list(set(interactions_df[interactions_df["personId"]==-1479311724257856983]["contentId"]))

cont=cont[~cont.contentId.isin(related_content)]
cont= cont.sort_values(by="score", ascending=False)[0:1000]
cont= cont.merge(articles_df, left_on="contentId", right_on="contentId", how="left")
cont.text


