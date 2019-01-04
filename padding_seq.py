#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 11:12:01 2019

@author: tarun.bhavnani@dev.smecorner.com
"""

#why pad sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_sequences([[1, 2, 3], [3, 4, 5, 6], [7, 8]])
#zeros added to the front

#If you rather want to pad to the end of the sequences you can set padding='post'.
pad_sequences([[1, 2, 3], [3, 4, 5, 6], [7, 8]], padding='post')

#If you want to specify the maximum length of each sequence you can use the maxlen argument. This will truncate all sequences longer than maxlen.

pad_sequences([[1, 2, 3], [3, 4, 5, 6], [7, 8]], maxlen=3)

#see it takes the last ones

pad_sequences([[1, 2, 3], [3, 4, 5, 6], [7, 8]], maxlen=3, padding='post')
#its takes first