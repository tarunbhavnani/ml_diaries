# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:30:21 2024

@author: tarun
"""

#two pointer technique
#Given a sorted array A (sorted in ascending order), having N integers, find if there exists any pair of 
#elements (A[i], A[j]) such that their sum is equal to X.

import random

lst=[random.randint(0,100) for i in range(20)]
lst=sorted(lst)

x=70


#basic O(log(N^2)) approach
for num, i in enumerate(lst):
    for j in lst[num+1:]:
        if i+j==x:
            print(i,j)
        

#two pointer approch

#one pointer at 0 , one at n-1
#since sorted, lowest on left, we add the two if less than value we move left pointer, if gresater than value we move right one


i=0
j=len(lst)-1

#iterate till i!=j

while i<j:
    if lst[i]+lst[j]==x:
        print(lst[i], lst[j])
        i+=1
        j-=1
    
    elif lst[i]+lst[j]<x:
        i+=1
    
    elif lst[i]+lst[j]>x:
        j-=1

