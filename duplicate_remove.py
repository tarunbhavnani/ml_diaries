#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:27:10 2019

@author: tarun.bhavnani
https://towardsdatascience.com/de-duplicate-the-duplicate-records-from-scratch-f6e5ad9e79da
"""
#!pip install sparse_dot_topn
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

df = pd.read_csv( '/home/tarun.bhavnani/Desktop/hotels.csv', encoding="latin-1")

[[(j,k) for j,k in i.split(',"')] for i in df['name,address'] ]
[i for i in df['name,address']]

"""
df['name']=0
df['address']=0
for i in range(0, len(df)):
    if df['name,address'].iloc[i][0]=='"':
        df['name,address'].iloc[i]=df['name,address'].iloc[i][1:]
        
    df['name'].iloc[i]=df['name,address'].iloc[i].split('"')[0]
    df['address'].iloc[i]=df['name,address'].iloc[i].split('"')[1]

df['name']=df['name,address'].split('"')[0]
"""

df['name']=0
df['address']=0
for i in range(0, len(df)):
        
    df['name'].iloc[i]=df['name,address'].iloc[i].split(',')[0]
    df['address'].iloc[i]=",".join(df['name,address'].iloc[i].split(',')[1:])


##
    
df['name_address'] = df['name'] + ' ' + df['address']
name_address = df['name_address']
vectorizer = TfidfVectorizer("char", ngram_range=(1, 4), sublinear_tf=True)
tf_idf_matrix = vectorizer.fit_transform(name_address)




import numpy as np
def awesome_cossim_top(A, B, ntop, lower_bound=0):
  
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape
 
    idx_dtype = np.int32
 
    nnz_max = M*ntop
 
    indptr = np.zeros(M+1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)

    ct.sparse_dot_topn(
        M, N, np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr, indices, data)

    return csr_matrix((data,indices,indptr),shape=(M,N))

matches = awesome_cossim_top(tf_idf_matrix, tf_idf_matrix.transpose(), 5)


def get_matches_df(sparse_matrix, name_vector, top=840):
    non_zeros = sparse_matrix.nonzero()
    
    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]
    
    if top:
        nr_matches = top
    else:
        nr_matches = sparsecols.size
    
    left_side = np.empty([nr_matches], dtype=object)
    right_side = np.empty([nr_matches], dtype=object)
    similairity = np.zeros(nr_matches)
    
    for index in range(0, nr_matches):
        left_side[index] = name_vector[sparserows[index]]
        right_side[index] = name_vector[sparsecols[index]]
        similairity[index] = sparse_matrix.data[index]
    
    return pd.DataFrame({'left_side': left_side,
                          'right_side': right_side,
                           'similarity': similairity})

matches_df = get_matches_df(matches, name_address)


#we want ony the top matches
hj=matches_df[matches_df['similarity'] < 0.99999][matches_df['similarity'] >.5].sort_values(by=['similarity'], ascending=False).head(30)


kl=matches_df[matches_df['similarity'] < 0.50].right_side.nunique()
