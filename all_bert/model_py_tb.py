# -*- coding: utf-8 -*-
"""
Created on Tue May 11 10:58:22 2021

@author: ELECTROBOT
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
import re

dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data
target= dataset.target

def clean(doc):
    doc= re.sub(r'[^A-Za-z0-9\.,?\' ]', " ", doc)
    doc= re.sub(r'\s+', " ", doc)
    return doc

documents= [clean(i) for i in documents]

data= [(i,j) for i,j in zip(documents, target)]
num_train=int(.9*len(data))
#num_test= int(.1*len(data))

from torch.utils.data.dataset import random_split
train, test=random_split(data,[num_train,len(data)-num_train])


# =============================================================================
# 
# =============================================================================



tokenizer = get_tokenizer('basic_english')
from torchtext.vocab import Vocab
from collections import Counter
#build vocab
counter= Counter()
for (text, label) in train:
    counter.update(tokenizer(text))

vocab=Vocab(counter, min_freq=1)

text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
label_pipeline = lambda x: int(x) - 1
    






def collate_batch(batch, max_len=300):
    label_list, text_list=[],[]
    for (_text, _label) in batch:
        
         label_list.append(label_pipeline(_label))
         #processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
         processed_text = text_pipeline(_text)
         if len(processed_text)>=max_len:
             processed_text= processed_text[0:max_len]
         else:
            processed_text= processed_text+[0]*(max_len-len(processed_text))

         text_list.append(processed_text)
         
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = torch.tensor(text_list, dtype=torch.long)
    
    
    #text_list = [torch.tensor(i, dtype=torch.int64) for i in text_list]
    return text_list, label_list



from torch.utils.data import DataLoader

train_loader= DataLoader(train, batch_size=64, shuffle=True, collate_fn=collate_batch)
test_loader= DataLoader(test, batch_size=64, shuffle=True, collate_fn=collate_batch)

asd=next(iter(train_loader))






# =============================================================================
# Lets create a pytorch model and visit data loader agian later 
# =============================================================================

from torch import nn

class text_classification_tb(nn.Module):
    def __init__(self, vocab_size,embedding_dim, embedding_matrix, out_channel, max_len, num_labels):
        super(text_classification_tb, self).__init__()
        self.embedding_layer= nn.Embedding(vocab_size, embedding_dim)
        #self.embedding_layer.weight= nn.parameter(embedding_matrix,requires_grad=False)
        
        self.conv_layer= nn.Conv1d(embedding_dim, out_channel, kernel_size=2)
        self.rel= nn.ReLU()
        self.pool = nn.MaxPool1d(2, 2)
        self.fc= nn.Linear(out_channel*(max_len-1), num_labels)
        
        
    def forward(self,text):
        output= self.embedding_layer(text)
        output = output.permute(0, 2, 1)
        output = self.conv_layer(output)
        output= self.rel(output)

        output = output.view(output.size(0),-1)
        output=  self.fc(output)
        return output
        

model= text_classification_tb(vocab_size,embedding_dim=100, embedding_matrix=0, out_channel=64, max_len=300, num_labels=len(set(target)))




x = torch.empty(batch_size, sentence_len, dtype=torch.long).random_(vocab_size)
model.train()
#model.__call__(x)
#model.forward(x)
model(x)



# =============================================================================
# create a dataloader which gives a batch of tokenized texts and labels, all torch tesnors
# =============================================================================

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)


model.train()
text_batch, label_batch=next(iter(train_loader))
predited_label=model(text_batch)
loss = criterion(predited_label, label_batch)




            
    
    















