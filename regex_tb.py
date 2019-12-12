#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:31:34 2019

@author: tarun.bhavnani
"""

#extract from between two "abc" and "xyz"

re.search('abc(.*)xyz', text).group(1)

#we can use for < and > also

text= "it is <see1> ths is what it <gjj>"
re.search('<(.*)>', text)


not wori\kinf if more than one pair as takes the first and last only



#multiple delimiters

delimiters = "a", "...", "(c)"

example = "stackoverflow (c) is awesome... isn't it?"

regexPattern = '|'.join(map(re.escape, delimiters))

regexPattern
#'a|\\.\\.\\.|\\(c\\)'

re.split(regexPattern, example)
['st', 'ckoverflow ', ' is ', 'wesome', " isn't it?"]


#extract digits
txt="Re: DISBURSEMENT KIT - DMI - 1041442 - SHREE G"
re.findall(r'\d+', txt)
re.findall(r'\b(\d{7})\b', txt)
