#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 17:16:49 2019

@author: tarun.bhavnani
"""
import os
import pandas as pd
from collections import Counter
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype



os.chdir('/home/tarun.bhavnani/Desktop/Projects/WOE-and-IV')
df= pd.read_csv('raw_binned_data.csv')

df=df.drop(['Unnamed: 0','Loan.refrence.number'], axis=1)
df['target'] = df['Bounced'].apply(lambda x : 1 if x == 'B' else 0)  # Convert to numeric
df = df.drop('Bounced',axis=1)


#remove null and nan:
#check
[(i,sum(df[i].isnull())) for i in df if sum(df[i].isnull())>0]
#[('Name', 15), ('industry', 228), ('constitution', 3)]

for i in df:
    print(i)
    if is_string_dtype(df[i])==True:
        df[i][df[i].isnull()]="Unknown"
    elif is_numeric_dtype(df[i])==True:
        df[i][df[i].isnull()]=0
    
    
#check
[(i,sum(df[i].isnull())) for i in df if sum(df[i].isnull())>0]
#[]

#cats=[i for i in df if is_string_dtype(df[i])==True and len(set(df[i]))<100 and len(set(df[i]))>1 ]
#[len(set(df[i])) for i in df if is_string_dtype(df[i])==True]

#nums=[i for i in df if is_numeric_dtype(df[i])==True and len(set(df[i]))>1]

#fdf=df[cats+nums]

#call from woe.py
final_iv, IV = data_vars(df,df.target)


#fiv=final_iv[final_iv['COUNT']>15]
#fiv=fiv[fiv['EVENT_RATE']>.1]

##

transform_vars_list = df.columns.difference(['target'])
transform_prefix = 'new_' # leave this value blank if you need replace the original column values

for var in transform_vars_list:
    print(var)
    small_df = final_iv[final_iv['VAR_NAME'] == var]
    transform_dict = dict(zip(small_df.MAX_VALUE,small_df.WOE))
    replace_cmd = ''
    replace_cmd1 = ''
    for i in sorted(transform_dict.items()):
        replace_cmd = replace_cmd + str(i[1]) + str(' if x <= ') + str(i[0]) + ' else '
        replace_cmd1 = replace_cmd1 + str(i[1]) + str(' if x == "') + str(i[0]) + '" else '
    replace_cmd = replace_cmd + '0'
    replace_cmd1 = replace_cmd1 + '0'
    if replace_cmd != '0':
        try:
            df[transform_prefix + var] = df[var].apply(lambda x: eval(replace_cmd))
        except:
            df[transform_prefix + var] = df[var].apply(lambda x: eval(replace_cmd1))



