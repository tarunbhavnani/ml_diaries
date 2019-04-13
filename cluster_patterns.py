#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 12:30:38 2019

@author: tarun.bhavnani@dev.smecorner.com
"""


import pandas as pd
import os
os.chdir("/home/tarun.bhavnani@dev.smecorner.com/Desktop/ocr_trans")

fdf=pd.read_csv("finaldf8.csv")
#cluetsring for patterns

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

fdf["counter"].value_counts()
#fdf["counter"].value_counts()

dat=fdf[fdf["counter"]==338]
#dat=fdf[fdf["counter"]==338]

#Debit transactions

def pattern(dat):
  
  def sortSecond(val): 
      return val[1]  

  
  dfg=pd.DataFrame( columns =['Debits', 'Credits']) 
  
  narrations = list(dat["Des"][dat["Credit"]==0])
  #jk=narrations
  #remove stopwords, to improve results
  stopwords= ["neft", "rtgs","imps","cash", "cheque", "tpt", "transfer", "to"]
  narrations=[" ".join([j for j in i.split() if j not in stopwords]) for i in narrations]
  list_groups_debit = []
  group_count = 0

  for i,string in enumerate(narrations):
   #if group_count>1000:
   #  break
   #else:
    #print(i,string)
    try:
    
      match_list = process.extract(string, narrations, scorer = fuzz.token_set_ratio, limit = len(narrations))     
      match_list = [ele[0] for ele in match_list if ele[1] >80]

      if len(match_list) > 5: #in list(range(2, 10)):
          list_groups_debit.append(tuple((match_list, len(match_list), group_count)))
          print(group_count)
          group_count +=1


      #for ele in match_list:
      #  narrations.remove(ele)
      narrations=[i for i in narrations if i not in match_list]
        #print("a")
      print("cccccccccccccc")
    except:
      print("skipping",i,string)

 
  list_groups_debit.sort(key = sortSecond,reverse = True)
  

  dfg["Debits"]=pd.Series([(i[0][1],i[1]) for i in list_groups_debit[0:10]])
  #dfg["Debits-Counts"]=pd.Series([i[1] for i in list_groups_debit[0:10]])



  narrations = list(dat["Des1"][dat["Debit"]==0])
  list_groups_credit = []
  group_count = 0

  
  for i,string in enumerate(narrations):
    try:
    #print(i,string)
      match_list = process.extract(string, narrations, scorer = fuzz.token_set_ratio, limit = len(narrations))     
      match_list = [ele[0] for ele in match_list if ele[1] >80]
      if len(match_list) > 5: #in list(range(2, 10)):
          list_groups_credit.append(tuple((match_list, len(match_list), group_count)))
          print(group_count)
          group_count +=1


      #for ele in match_list:
      #  narrations.remove(ele)
      narrations=[i for i in narrations if i not in match_list]
      #print("a")
      #print("cccccccccccccc")
    except:
      print("skipping",i,string)

#  def sortSecond(val): 
#      return val[1]  

  list_groups_credit.sort(key = sortSecond,reverse = True)

  #dfg["Credits"]=[i[0][1] for i in list_groups_credit[1:10]]
  dfg["Credits"]=pd.Series([(i[0][1],i[1]) for i in list_groups_credit[0:10]])
  #gh=[i[0][1] for i in list_groups_credit[1:10]]
  #[y if y else i for i,y in zip(dfg["Credits"],gh)]
  
  return(dfg)

fgh= pattern(dat)




