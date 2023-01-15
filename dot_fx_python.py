# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 16:29:06 2022

@author: ELECTROBOT
"""

import numpy as np


np.array([])

hj=np.random.rand(10).reshape(5,2)
hj.shape
hk= np.random.rand(8).reshape(4,2)

hj.dot(hk).shape


hj
def dot_fx(mat1, mat2):
    
    if mat1.shape[1]==mat2.shape[0]:
        final_mat=np.zeros(mat1.shape[0]* mat2.shape[1]).reshape(mat1.shape[0], mat2.shape[1])
        
        for num1, i in enumerate(mat1):
            #break
            for  num2,j in enumerate(mat2.T):
                #break
                res=0
                for x in range(len(i)):
                    res+=i[x]*j[x]
                final_mat[num1,num2]=res
        return final_mat
    else:
        print("matrix shapes not alligned")
                
        

