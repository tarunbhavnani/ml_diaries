#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 12:43:22 2019

@author: tarun.bhavnani@dev.smecorner.com
"""

#loading and classifing transaction analysis

import os
os.chdir("/home/tarun.bhavnani/Desktop/ocr_trans/Final_Models/new")


from transc_function import clean_transc, workspace


fg= workspace("405201907_Final.xlsx")

fg.to_csv("final_data.csv")



#list(dat1)


"""
#left:

int_coll--> further segregate if debit or credit
nach/emi--> further segregate, if debit then emi, if credit then disbursal


"""


