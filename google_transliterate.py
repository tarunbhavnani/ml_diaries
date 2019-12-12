#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 09:50:33 2019

@author: tarun.bhavnani

"""

from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
   

data = 'idam adbhutam'

print(transliterate(data, sanscript.HK, sanscript.TELUGU))
#ఇదమ్ అద్భుతమ్

print(transliterate(data, sanscript.ITRANS, sanscript.DEVANAGARI))
#इदम् अद्भुतम्

scheme_map = SchemeMap(SCHEMES[sanscript.VELTHUIS], SCHEMES[sanscript.TELUGU])

print(transliterate(data, scheme_map=scheme_map))
#ఇదమ్ అద్భుతమ్


name= "arun agarwal"
asd=transliterate(name, sanscript.ITRANS, sanscript.DEVANAGARI)

back_name=transliterate(asd, sanscript.DEVANAGARI, sanscript.ITRANS)


name= "mera naam tarun hai "

for i in range(0,100):
    asd=transliterate(name, sanscript.ITRANS, sanscript.DEVANAGARI)
    print(asd)

    name=transliterate(asd, sanscript.DEVANAGARI, sanscript.ITRANS)
    print(name)

    
