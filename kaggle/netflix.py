# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:45:18 2023

@author: ELECTROBOT
"""

import os
import pandas as pd

movie= pd.read_csv(r"C:\Users\ELECTROBOT\Desktop\kaggle\netflix\movie_titles.csv",encoding='latin-1',on_bad_lines='skip', header=None)


# temp= pd.read_csv(r"C:\Users\ELECTROBOT\Desktop\kaggle\netflix\combined_data_1.txt", sep=",")
# temp=temp.reset_index()
# temp.columns=["item", "rating", "date"]


fdf=pd.DataFrame()
df1 = pd.read_csv(r"C:\Users\ELECTROBOT\Desktop\kaggle\netflix\combined_data_1.txt", header = None, names = ['Cust_Id', 'Rating', "Date"], usecols = [0,1,2])

fdf=pd.concat([fdf, df1], axis=0)
del df1

# df1 = pd.read_csv(r"C:\Users\ELECTROBOT\Desktop\kaggle\netflix\combined_data_2.txt", header = None, names = ['Cust_Id', 'Rating', "Date"], usecols = [0,1,2])

# fdf=pd.concat([fdf, df1], axis=0)
# del df1
# df1 = pd.read_csv(r"C:\Users\ELECTROBOT\Desktop\kaggle\netflix\combined_data_3.txt", header = None, names = ['Cust_Id', 'Rating', "Date"], usecols = [0,1,2])

# fdf=pd.concat([fdf, df1], axis=0)
# del df1
# df1 = pd.read_csv(r"C:\Users\ELECTROBOT\Desktop\kaggle\netflix\combined_data_4.txt", header = None, names = ['Cust_Id', 'Rating', "Date"], usecols = [0,1,2])

# fdf=pd.concat([fdf, df1], axis=0)
# del df1

fdf= fdf.reset_index(drop=True)



len(fdf.Cust_Id)
#100498277
len(set(fdf.Cust_Id))
#497959

import numpy as np
p = fdf.groupby('Rating')['Rating'].agg(['count'])
# get movie count
movie_count = fdf.isnull().sum()[1]
# get customer count
cust_count = fdf['Cust_Id'].nunique() - movie_count
# get rating count
rating_count = fdf['Cust_Id'].count() - movie_count


df_nan = pd.DataFrame(pd.isnull(fdf.Rating))
df_nan = df_nan[df_nan['Rating'] == True]
df_nan = df_nan.reset_index()

movie_np = []
movie_id = 1

for i,j in zip(df_nan['index'][1:],df_nan['index'][:-1]):
    # numpy approach
    temp = np.full((1,i-j-1), movie_id)
    movie_np = np.append(movie_np, temp)
    movie_id += 1


# Account for last record and corresponding length
# numpy approach
last_record = np.full((1,len(fdf) - df_nan.iloc[-1, 0] - 1),movie_id)
movie_np = np.append(movie_np, last_record)

# remove those Movie ID rows
fdf = fdf[pd.notnull(fdf['Rating'])]

fdf['Movie_Id'] = movie_np.astype(int)
fdf['Cust_Id'] = fdf['Cust_Id'].astype(int)

for num,i in enumerate(range(0,len(fdf), round(len(fdf)/10))):
    print(num)    
    temp=fdf.iloc[i: i+round(len(fdf)/10)]
    temp.to_csv(f"netflix_{num}.csv", index=False)
    

#lets take 100 k records

df= fdf.iloc[0:100000]

list(fdf)
tmp=fdf.groupby("Cust_Id")["Rating"].count().reset_index()
cust= list(tmp[tmp.Rating>1500]["Cust_Id"])

tmp=fdf.groupby("Movie_Id")["Rating"].count().reset_index()
mov= list(tmp[tmp.Rating>800]["Movie_Id"])

fdf_=fdf[fdf.Cust_Id.isin(cust)]
fdf_=fdf_[fdf_.Movie_Id.isin(mov)]




reviewmatrix = fdf_.pivot(index="Cust_Id", columns="Movie_Id", values="Rating").fillna(0)
matrix = reviewmatrix.values

# from sklearn.decomposition import TruncatedSVD
# svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
# svd.fit(matrix)


# print(svd.explained_variance_ratio_)
# print(svd.explained_variance_ratio_.sum())
# print(svd.singular_values_)



# def cosine_similarity(v,u):
#     return (v @ u)/ (np.linalg.norm(v) * np.linalg.norm(u))

# highest_similarity = -np.inf
# highest_sim_col = -1
# for col in range(1,vh.shape[1]):
#     similarity = cosine_similarity(vh[:,0], vh[:,col])
#     if similarity > highest_similarity:
#         highest_similarity = similarity
#         highest_sim_col = col



import numpy as np
from scipy.sparse.linalg import svds


num_components = 2
u, s, v = svds(matrix, k=num_components)

u.shape, s.shape, v.shape

X = u.dot(np.diag(s))  # output of TruncatedSVD



test_id= matrix[0,:]
tmp=v.dot(test_id)  # output of TruncatedSVD
pred=tmp.dot(v) 
pred=[round(i) for i in pred]
list(test_id)

from sklearn.metrics import f1_score, accuracy_score
accuracy_score(pred, list(test_id))
f1_score(pred, list(test_id), average="micro")






















