# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 10:53:28 2021

@author: ELECTROBOT
"""

import torch
import pandas as pd
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split
#import spacy
#nlp= spacy.load('en_core_web_sm')
os.chdir(r'C:\Users\ELECTROBOT\Desktop\kaggle\commonlit_readbility')
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



train= pd.read_csv('train_june13.csv')

import torch
from transformers import BertTokenizer, BertModel, BertConfig
model_path = r'C:\Users\ELECTROBOT\Desktop\bert_base_uncased'
tokenizer= BertTokenizer.from_pretrained(model_path)

bert_model= BertModel.from_pretrained(model_path)
bert_model.config.output_hidden_states=True

#train_tokens= tokenizer.encode_plus(train.excerpt.iloc[0],max_length=512,pad_to_max_length=True,truncation=True)
train_tokens= tokenizer.encode_plus(train.excerpt.iloc[0])#,output_hidden_states=True)


input_ids=torch.tensor(train_tokens['input_ids']).view(1,-1)
attention_mask=torch.tensor(train_tokens['attention_mask']).view(1,-1)
#label= torch.tensor(train.target.iloc[0]).unsqueeze(0)


#bert_model.config.num_labels=1
bert_output= bert_model(input_ids, attention_mask=attention_mask)
bert_output= bert_model(input_ids, attention_mask=attention_mask, output_attentions=True)




bert_output
bert_output[0].shape
bert_output[1].shape
bert_output[0].view(220,768)
bert_output[0][0]==bert_output[0].view(220,768)

bert_output[0][:,1,:]

bert_output[1].shape


