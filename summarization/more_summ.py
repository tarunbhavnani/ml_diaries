# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 16:44:57 2021

@author: tarun
"""
import re
import os
os.chdir(r'C:\Users\tarun\Desktop\summarization_bart')
import fitz


#doc=fitz.open('rf.pdf')
doc=fitz.open('1810.09305.pdf')

all_text=[]
for page in doc:
    text=page.getText().encode('utf8')
    text= text.decode('utf8')
    all_text.append(text)


all_text= "\n".join([i for i in all_text])

all_text=re.sub('\n', ' ', all_text)
all_text= re.sub(r'\s+', " ", all_text)

print(all_text)

text_blob=re.sub(r'(?<=\[).+(?=\])', "", all_text)
text_blob=re.sub(r'(?<=\().+(?=\))', "", text_blob)

#remove html etx basic stuff


text_blob=re.sub("[\(\[].*?[\)\]]", "", all_text)
text_blob= re.sub(r'[^A-za-z0-9.\' ]', " ", text_blob)


text_blob= re.sub(r'\s+', " ", text_blob)

#split sentences and validate


all_sents= split_into_sentences(text_blob)

import spacy
nlp= spacy.load('en_core_web_sm')

all_sentences=valid(all_sents)

all_sentences=[i for i in all_sentences if valid_toggle(re.sub(r'[^A-Za-z ]'," ",i))<.4]

text_blob= " ".join([i for i in all_sentences])

# =============================================================================
# 
# =============================================================================
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
                 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
                 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
                 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
                 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
                 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
                 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
                 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should',
                 'now']
#Frequency of words
words={}
for sent in all_sentences:
    for word in sent.lower().split():
        if word not in stopwords:
            if word not in words:
                words[word]=1
            else:
                words[word]+=1

maxi=max([j for i,j in words.items()])
import pandas as pd

weighted_freq=pd.DataFrame(words.items(), columns=['word', "freq"])
weighted_freq['wt']=[i/maxi for i in weighted_freq.freq]

wt_fr={i:j for i,j in zip(weighted_freq.word, weighted_freq.wt)}
# =============================================================================
# sentence weights
# =============================================================================

#[i.lower().split() for i in all_sentences]

def weight(sent):
    
    return sum([wt_fr[token] if token in wt_fr else 0 for token in sent.lower().split()])
    
sent_weights=[weight(sent) for sent in all_sentences]

fdf= pd.DataFrame({"sentence": all_sentences, "weights":sent_weights})

fdf['weights'].quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])


fdf1= fdf[fdf.weights>2.3]


text_blob=" ".join([i for i in fdf1.sentence])

summary= summarize(text_blob, tokenizer, model)









    
    
    













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



