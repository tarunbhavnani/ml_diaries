#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 17:55:43 2019

@author: tarun.bhavnani@dev.smecorner.com
"""

def remove_num(tt):
 
 if len(re.findall('\(number\)',  tt, flags=re.IGNORECASE))>0:
   tt=re.sub('\(number\)',"",  tt, flags=re.IGNORECASE)
   tt=re.sub('\[',"", tt, flags=re.IGNORECASE)
   re.sub('\]',"", tt, flags=re.IGNORECASE)
 return tt

def remove_loc(tt):
 
 if len(re.findall('\(location\)',  tt, flags=re.IGNORECASE))>0:
   tt=re.sub('\(location\)',"",  tt, flags=re.IGNORECASE)
   tt=re.sub('\[',"", tt, flags=re.IGNORECASE)
   re.sub('\]',"", tt, flags=re.IGNORECASE)
 return tt
