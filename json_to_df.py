#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 17:39:53 2019

@author: tarun.bhavnani@dev.smecorner.com
"""

import json
 


pd.read_json(_, orient='index')
 
 

import os
os.chdir("/home/tarun.bhavnani@dev.smecorner.com/Desktop/ocr_trans")
file='All_transaction_Old_OCR FILES.json'


json = json.loads(open(file).read())

        
#The MongoDB JSON dump has one object per line, so what worked for me is:


import json    

data = []
with open(file) as f:
    for line in f:
        data.append(json.loads(line))


data[1].index
data[0].keys()
[i for i in data[0].values()]
[i for i in data[0].keys()]





with open(file) as f:
    for line in f:
      asd= json.loads(line)
        data.append(json.loads(line))





Check this snip out.

# reading the JSON data using json.load()
#file = 'data.json'
with open(file) as train_file:
    dict_train = json.load(train_file)

# converting json dataset from dictionary to dataframe
train = pd.DataFrame.from_dict(asd)
train.reset_index(level=0, inplace=True)
Hope it helps :)

  
import json  
from pandas.io.json import json_normalize
df = pd.DataFrame.from_dict(json_normalize(data), orient='columns')
df.head()






######################3
Creating dataframe from dictionary object.

import pandas as pd
data = [{'name': 'vikash', 'age': 27}, {'name': 'Satyam', 'age': 14}]
df = pd.DataFrame.from_dict(data, orient='columns')

df
Out[4]:
   age  name
0   27  vikash
1   14  Satyam
If you have nested columns then you first need to normalize the data:

from pandas.io.json import json_normalize
data = [
  {
    'name': {
      'first': 'vikash',
      'last': 'singh'
    },
    'age': 27
  },
  {
    'name': {
      'first': 'satyam',
      'last': 'singh'
    },
    'age': 14
  }
]

df = pd.DataFrame.from_dict(json_normalize(data), orient='columns')

df    
Out[8]:
age name.first  name.last
0   27  vikash  singh
1   14  satyam  singh
Source: https://github.com/vi3k6i5/pandas_basics/blob/master/1_a_create_a_dataframe_from_dictonary.ipynb






