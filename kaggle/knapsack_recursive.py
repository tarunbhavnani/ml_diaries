# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 11:15:52 2024

@author: tarun
"""

names = ['wine', 'beer', 'pizza', 'burger', 'fries',
         'cola', 'apple', 'donut', 'cake']
values = [89,90,95,100,90,79,50,10]
calories = [123,154,258,354,365,150,95,195]

#value is how much you value that item, but you dont want to consume more than 750 cals!
#requirement: get max value inb 750 cals or less
#solve!!




items= [(i,(j,k)) for i,j,k in zip(names, values, calories)]


# =============================================================================
# # lets design a recursive function to solve this. its not bfs and dfs, its recursive
# =============================================================================

def rec(items, available):
    # Base case: If there are no more items to consider, return a value of 0.
    if len(items) == 0:
        return 0
    
    # Check if the first item's weight exceeds the available capacity.
    if items[0][1][1] > available:
        # If the item is too heavy, skip it and proceed with the remaining items.
        return rec(items[1:], available)
    
    # Otherwise, consider the first item in the list.
    # Calculate the result when including the first item:
    with_value = rec(items[1:], available - items[0][1][1]) + items[0][1][0]
    
    # Calculate the result when excluding the first item:
    without_value = rec(items[1:], available)
    
    # Compare the results of including vs. excluding the first item:
    return max(with_value, without_value)       


#get value and items chosen


available=750
def rec_tree(items, available):
    
    #define the stop
    
    if len(items)==0:
        result= (0,())
        
        #check if the first element is higher than available, if itb is remove it
    elif items[0][1][1]>available:
        result= rec_tree(items[1:], available)
    
    else:
        #check if taking the first one improves things or not
        
        next_item= items[0]
        
        with_value,selected_with = rec_tree(items[1:], available-next_item[1][1])
        with_value+=next_item[1][0]
        
        without_value, selected_without=rec_tree(items[1:], available)
        print(selected_without)
        
        if with_value>without_value:
            result= (with_value,  selected_with+next_item)
        else:
            result= (without_value,selected_without)            
    return result
    
            
            
            
rec_tree(items=items, available=750)       
            
    
# =============================================================================
# there can be a greedy approach as well
# =============================================================================
    


def greedy(items, available):
    
    items=sorted(items, key= lambda x: x[1][0])
    items=sorted(items, key= lambda x: x[1][1])
    
    #val per cal wise
    #items=[(i,(j[0],j[1], j[0]/j[1])) for i,j in items]
    #items=sorted(items, key= lambda x: x[1][2], reverse=True)
    
    
    #val wise
    items=sorted(items, key= lambda x: x[1][0], reverse=True)
    
    
    #calwise
    #items=sorted(items, key= lambda x: x[1][1], reverse=True)
    
    
    result = []
    totalValue=0
    for item in items:
        #print(item)
        #break
        if item[1][1]<=available:
            result.append(item)
            available-=item[1][1]
            totalValue+=item[1][0]
        else:
            pass
            
            
    return (result, totalValue)
    












        