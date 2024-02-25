# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 18:30:46 2024

@author: tarun
"""

import os
os.chdir(r'D:\kaggle\bank churn dataset')
import pandas as pd

train= pd.read_csv('train.csv')
test= pd.read_csv('test.csv')
ss= pd.read_csv('sample_submission.csv')


train.Exited.value_counts(normalize=True)

