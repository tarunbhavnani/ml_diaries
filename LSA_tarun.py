#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:28:38 2019

@author: tarun.bhavnani@dev.smecorner.com
http://mccormickml.com/2016/03/25/lsa-for-text-classification-tutorial/
"""

import pandas as pd
dt= pd.read_csv("/home/tarun.bhavnani@dev.smecorner.com/Desktop/git_tarun/Reviews.csv")
dt.head()
list(dt)
dt.Summary[0]
dt.Text[0]


#in lsa we create the vector representation of the doc.
#thuis we can find similarity among docs.
"""you just use SVD to perform dimensionality reduction on the tf-idf vectors–that’s really all there 
is to it! """


