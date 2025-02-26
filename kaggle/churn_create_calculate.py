# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 13:23:54 2024

@author: tarun
"""

import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

df= pd.read_csv(r"C:\Users\tarun\Desktop\MIT-Optmization\Churn\data.csv", encoding="ISO-8859-1")

df_=df.head()

#lets ignoore the returns

temp=df[df.Quantity<0]

df= df[df.Quantity>0]

list(df)

df.InvoiceDate= pd.to_datetime(df.InvoiceDate)

df['my']=[str(i.year)+"-"+str(i.month) for i in df.InvoiceDate]
df['total_price']=df['Quantity']*df['UnitPrice']

#bunch by invoce num
list(df)
inv= df.groupby('InvoiceNo').agg({'StockCode':lambda x:list(x),
                                  'Quantity': lambda x: list(x),
                                 'InvoiceDate': lambda x: list(set(x))[0],
                                 'UnitPrice': lambda x: list(x),
                                 'CustomerID': lambda x: list(set(x))[0],
                                 'Country': lambda x: list(set(x))[0],
                                 'my': lambda x: list(set(x))[0],
                                 'total_price': lambda x: sum(x)}).reset_index()


see=inv.groupby(['CustomerID','my']).agg({'Quantity':lambda x: list(x),
                                          'total_price':lambda x: sum(x),
                                          'InvoiceNo':lambda x: list(x) }).reset_index()

df_=df.groupby(['CustomerID', 'my']).agg({'Quantity':sum, "total_price":sum}).reset_index()


#get churn variable
#def: lets consider all the cuistomers who are there in month 1 and not there in month2 and 3 to be churned
# later we can refine it to days

# here we will find customers whoi have churned and come back in that case we will consider them as new customers



months= {i:j for i,j in enumerate(sorted(set(df_.my)))}

final_data=pd.DataFrame()

for month in months:
    try:
        temp= df_[df_.my==months[month]] 
        next1= df_[df_.my==months[month+1]]
        next2=df_[df_.my==months[month+2]]
        
        customers= set(list(temp[~temp.CustomerID.isnull()].CustomerID))
        
        # Convert list2 and list3 to sets
        set2 = set(list(next1[~next1.CustomerID.isnull()].CustomerID))
        set3 = set(list(next2[~next2.CustomerID.isnull()].CustomerID))
        
        # Find the union of set2 and set3
        union_set = set2.union(set3)
        
        # Find the difference between list1 and the union of list2 and list3
        churned = [item for item in customers if item not in union_set]
        
        
        
        total_data= pd.concat([temp, next1, next2], axis=0)
        
        total_data=total_data[~total_data.CustomerID.isnull()]
        
        total_data['churned']= [1 if i in churned else 0  for i in total_data.CustomerID]
        
        #give new customer ids
        
        temp_ids={j:str(i)+"-"+str(month) for i,j in enumerate(set(total_data.CustomerID))}
        
        total_data["CustomerID"]=[temp_ids[i] for i in total_data.CustomerID]
        
        
        final_data=pd.concat([final_data, total_data], axis=0)
    
    except Exception as e:
        print(str(e))
        
    
final_data.churned.value_counts()
    
    
    
    
    
    
    





