#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 11:59:05 2018

@author: tarun.bhavnani@dev.smecorner.com
"""

import os
os.getcwd()
os.chdir('/home/tarun.bhavnani@dev.smecorner.com/Desktop/rasa_final_start')

os.listdir()

import pandas as pd
df=pd.read_excel( 'data_los.xlsx')
list(df)



names = ['aet2000','ppt2000', 'aet2001', 'ppt2001']
[i for i in filter(lambda x:'aet' in x, names)]


import re

df_app=df[[i for i in filter(lambda x: re.search(r'app',x),list(df))]]
df.iloc[:,1]


list(df[[i for i in filter(lambda x: re.search(r'phone',x),list(df))]])

df[[i for i in filter(lambda x: re.search(r'phone',x),list(df))]]
