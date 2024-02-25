# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:31:59 2024

@author: tarun
"""

import random

arr= [random.randint(0,100) for i in range(20)]
arr=[15, 97, 72, 53, 11, 43, 9, 73, 68, 66, 96, 72, 82, 8, 22, 2, 91, 46, 62, 46]


#bubble
#normal sort one by one

for i in range(len(arr)):
    for j in range(len(arr)-i-1):
        if arr[j]>arr[j+1]:
            arr[j], arr[j+1]=arr[j+1], arr[j]


# quick, pivot and conquer


def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]  # Choose the pivot element
    left = [x for x in arr if x < pivot]  # Elements less than the pivot
    middle = [x for x in arr if x == pivot]  # Elements equal to the pivot
    right = [x for x in arr if x > pivot]  # Elements greater than the pivot
    return quick_sort(left) + middle + quick_sort(right)    

        