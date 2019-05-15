#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:21:39 2019

@author: tarun.bhavnani@dev.smecorner.com
"""

from datetime import date, timedelta

d1 = date(2008, 8, 15)  # start date
d2 = date(2008, 8, 25)  # end date

holliday=[date(2008,8,17),date(2008,8,18),date(2008,8,19)]

         # timedelta
dt=[]
for i in range(delta.days + 1):
    dt.append(d1 + timedelta(i))
    
lent=0
for i in holliday:
  if i in dt:
    lent+=1
    
delta= d2-d1
final_delta= delta.days-lent