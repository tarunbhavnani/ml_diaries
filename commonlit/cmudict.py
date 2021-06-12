# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 13:53:52 2021

@author: ELECTROBOT
"""

import cmudict

def lookup_word(word_s):
    return cmudict.dict().get(word_s)        # standard dict access

def lookup2_word(word_s):
    entries = [e[1] for e in cmudict.entries() if e[0] == word_s]
    return entries

def count_syllables(word_s):
    count = 0
    phones = lookup_word(word_s)
    if phones:
        phones0 = phones[0]
        count = len([p for p in phones0 if p[-1].isdigit()])
    return count

word_s = 'hello'
word_s="uncertain"
phones = lookup_word(word_s)
phones2 = lookup2_word(word_s)
count = count_syllables(word_s)
print(f"PHONES({word_s!r}) yields {phones}\nCOUNT is {count}")
print(f"PHONES are same: {phones == phones2}")