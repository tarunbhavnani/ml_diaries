#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:14:35 2019

@author: tarun.bhavnani
"""

"""
NLU

intents and verbatins
a basic classification model based on lstms

"""

"""
Core

every utternace is an intent weather bot ir user

lets take "n" as the number of history we consider

then basically the bot will take th last 5 intents and predict the next intent

it is much like the seq2seq where you are predicting the next word

it can be a simple teacher forcing also as we did for language trabslations!



"""