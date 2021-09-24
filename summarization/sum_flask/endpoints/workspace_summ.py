# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 18:35:59 2021

@author: tarun
"""
import re
import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

model_path=r'C:\Users\tarun\Desktop\summarization_bart\model_files'
model, tokenizer= load_model(model_path)


summary= summarize(text_blob, tokenizer, model)

#doc_path=r"C:\Users\tarun\Desktop\data\bert_medium.pdf"
#doc=fitz.open(doc_path)
doc=fitz.open(os.path.join(session['Folder'], filename))
all_sentences=get_all_sents(doc)




text_blob= " ".join([i for i in all_sentences])


model_path=r'C:\Users\tarun\Desktop\summarization_bart\model_files'
model, tokenizer= load_model(model_path)
summary= summarize(text_blob, tokenizer, model)


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
words={}
for sent in all_sentences:
    for word in sent.lower().split():
        if word not in stopwords:
            if word not in words:
                words[word]=1
            else:
                words[word]+=1

maxi=max([j for i,j in words.items()])


weighted_freq=pd.DataFrame(words.items(), columns=['word', "freq"])
weighted_freq['wt']=[i/maxi for i in weighted_freq.freq]

wt_fr={i:j for i,j in zip(weighted_freq.word, weighted_freq.wt)}

def weight(sent):
   
    return sum([wt_fr[token] if token in wt_fr else 0 for token in sent.lower().split()])
   
sent_weights=[weight(sent) for sent in all_sentences]

fdf= pd.DataFrame({"sentence": all_sentences, "weights":sent_weights})

#thresh=fdf['weights'].quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

for quant in reversed([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]):
    thresh=fdf['weights'].quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])[quant]
    if len(" ".join([i for i in fdf[fdf.weights>thresh].sentence]).split())>1000:
        break


fdf1= fdf[fdf.weights>thresh]

text_blob=" ".join([i for i in fdf1.sentence])



