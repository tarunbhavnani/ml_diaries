# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 16:46:58 2021

@author: ELECTROBOT

"""

import re

    
    
def clean(text, html=False,brackets=False, digits=False):
    
    if html==True:
        text=re.sub(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});'," ", text)
    if brackets== True:
        text= re.sub(r'(?<=\[).+?(?=\])', "", text)
        text= re.sub(r'(?<=\().+?(?=\))', "", text)
    if digits== True:
        text= re.sub(r'\d+\.\d+|\d+', " ", text)
    
    text= re.sub(r'[^A-Za-z0-9\.\(\)\[\]@ ]', " ", text)
    text= re.sub(r'\s+', " ", text.strip())
    
    return text
    
    
def split_into_sentences(text):
    
    alphabets = "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr|No|no)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    decimals = "(\d*[.]\d*)"
    websites = "[.](com|net|org|io|gov|co|in)"
    decimals = r'\d\.\d'

    
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(decimals, lambda g: re.sub(r'\.', '<prd>', g[0]), text)
    # text= re.sub(r'(?<=\[).+?(?=\])', "", text) # remove everything inside square brackets
    # text= re.sub(r'(?<=\().+?(?=\))', "", text) # remove everything inside  brackets
    text = re.sub(r'\[(\w*)\]', "", text)  # remove evrything in sq brackets with sq brackets
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    if "i.e." in text: text = text.replace("i.e.", "i<prd>e<prd>")
    if "e.g." in text: text = text.replace("e.g.", "e<prd>g<prd>")
    if "e.g" in text: text = text.replace("e.g", "e<prd>g")
    if "www." in text: text = text.replace("www.", "www<prd>")
    text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
    if "”" in text: text = text.replace(".”", "”.")
    if "\"" in text: text = text.replace(".\"", "\".")
    if "!" in text: text = text.replace("!\"", "\"!")
    if "?" in text: text = text.replace("?\"", "\"?")
    
    text= re.sub("\w[.]\w", "<prd>", text) #replace dot in \w.\w by prd
    
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    # sentences = sentences[:-1]

    sentences = [s.strip() for s in sentences]
    
    return sentences



#def read(file):
    



# =============================================================================
# 
# =============================================================================
#for xml
text= clean(text, html=True)






    