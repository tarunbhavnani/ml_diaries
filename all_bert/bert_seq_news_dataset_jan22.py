# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 10:03:44 2022

@author: ELECTROBOT
"""

# =============================================================================
# #load libraries
# =============================================================================

from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import numpy as np
import torch
import re
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig

bert_model_path=r"C:\Users\ELECTROBOT\Desktop\model_dump\bert_base_uncased"


# =============================================================================
# #get data
# =============================================================================
dataset= fetch_20newsgroups(shuffle=True, random_state=1)

documents= dataset.data
target= dataset.target

# =============================================================================
# #clean documents
# =============================================================================

#hj=documents[1]
#re.split( "From:",hj, maxsplit=1)[1]


#re.search(r'From:(.*\n*.*)\w*:', hj).group(1)

#re.search(r'From:(.*?)Subject', hj).group(2)


# breaks=re.findall(r'\w*:', hj)
# final={}
# not_include= ['From:', 'Subject:', 'Re:', 'Lines:', 'Organization:']
# text=""
# for num, break1 in enumerate(breaks):
#     #break
#     if num==0:
#         hj1= re.split(break1, hj, maxsplit=1)
        
#         hj=hj1[1]
#     else:
#         hj1= re.split(break1, hj, maxsplit=1)
#         final[breaks[num-1]]=hj1[0]
#         if breaks[num-1] not in not_include:
#             text= text+" "+ hj1[0]
#         hj=hj1[1]
        




def clean_junk(text):
    text=re.sub(r'[A-Za-z0-9]*@[A-Za-z]*\.?[A-Za-z0-9]*', "", text)
    text=re.sub(r'http.*', " ", text)
    text=re.sub('<[^<]+?>', '', text)
    text=re.sub(r'(?<=\[).+?(?=\])', "", text) 
    text=re.sub(r'(?<=\().+?(?=\))', "", text) # remove everything inside  brackets
    text=re.sub(r'[^A-Za-z0-9 /.,: ]', " ", text)
    text=re.sub(r'\s+', " ", text)
    return text
def split_into_sentences(text):
    alphabets = "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr|No)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    decimals = "(\d*[.]\d*)"
    websites = "[.](com|net|org|io|gov|co|in)"
    decimals = r'\d\.\d'

    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(decimals, lambda g: re.sub(r'\.', '<prd>', g[0]), text)
    text= re.sub(r'(?<=\[).+?(?=\])', "", text) # remove everything inside square brackets
    text= re.sub(r'(?<=\().+?(?=\))', "", text) # remove everything inside  brackets
    text = re.sub(r'\[(\w*)\]', "", text)  # remove evrything in sq brackets with sq brackets
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    if "i.e." in text: text = text.replace("i.e.", "i<prd>e<prd>")
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
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    # sentences = sentences[:-1]

    sentences = [s.strip() for s in sentences]
    return sentences

def clean_text(hj):
    hj=split_into_sentences(clean_junk(hj))
    
    not_include= ['From:', 'Subject:', 'Re:', 'Lines:', 'Organization:']
    
    for ni in not_include:
        hj=[i for i in hj if ni not in i and len(i)>2]
    
    hj= " ".join([i for i in hj])
    return hj




sentences= [clean_text(i) for i in documents]




# =============================================================================
# #load tokenizer
# =============================================================================


tokenizer= BertTokenizer.from_pretrained(bert_model_path)

tokenizer.encode("this is a tokenizer")
# [101, 2023, 2003, 1037, 19204, 17629, 102]

tokenizer.encode_plus("this is a tokenizer")
#{'input_ids': [101, 2023, 2003, 1037, 19204, 17629, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1]}

tokenizer.encode_plus(["this is a tokenizer", "this is batch one"])
#{'input_ids': [101, 100, 100, 102], 'token_type_ids': [0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1]}

tokenizer.batch_encode_plus(["this is a tokenizer", "this is batch one"])
#{'input_ids': [[101, 2023, 2003, 1037, 19204, 17629, 102], [101, 2023, 2003, 14108, 2028, 102]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]}



# =============================================================================
# #define the data loader
# =============================================================================

#returns inpouts ids attention masks and respective labels

class loader(torch.utils.data.Dataset):
    def __init__(self, tokenizer, documents, labels= None, max_len=512):
        super().__init__()
        self.documents=documents
        
        if labels:
            self.labels= torch.tensor(labels)
        else:
            self.labels= None
        self.encoded= tokenizer.batch_encode_plus(self.documents,
                                                  padding="max_length",
                                                  max_length=max_len,
                                                  truncation=True,
                                                  return_attention_mask=True)    
        
        
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, index):
        input_ids= torch.tensor(self.encoded['input_ids'][index])
        attention_mask= torch.tensor(self.encoded['attention_mask'][index])
        if self.labels!=None:
            labels= self.labels[index]
            return input_ids, attention_mask, labels
        else:
            return input_ids, attention_mask
        


test
load= loader(tokenizer=tokenizer, documents=documents, labels= target.tolist())

kl=torch.utils.data.DataLoader(load, batch_size=4, shuffle=True)

for i in kl:
    break


# =============================================================================
# #create train and val data loaders
# =============================================================================


train,val, y_train, y_val= train_test_split(sentences, target, test_size=.3, stratify= target, shuffle=True)


train_loader= loader(tokenizer=tokenizer, documents=train, labels=y_train.tolist())
train_loader= torch.utils.data.DataLoader(train_loader, batch_size=4, shuffle=True)


val_loader= loader(tokenizer=tokenizer, documents=val, labels= y_val.tolist())
val_loader= torch.utils.data.DataLoader(val_loader, batch_size=4, shuffle=False )


# =============================================================================
# #define model optimizer and scheduler
# =============================================================================

bert_model= BertForSequenceClassification.from_pretrained(bert_model_path, num_labels= len(set(target)))
bert_model.num_labels


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bert_model.to(device)

params=list(bert_model.named_parameters())


optimizer = AdamW(bert_model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

epochs=1
from transformers import get_linear_schedule_with_warmup
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

from transformers import get_linear_schedule_with_warmup
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)




# =============================================================================
# #fine tune model
# =============================================================================

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


import time
t0=time.time()

best_valid_loss = float('inf')
train_losses=[]
val_losses=[]
val_accuracy=[]



for epoch in range(0,epochs):
    
    print("epoch---{}".format(epoch))
    
    total_train_loss=0
    
    bert_model.train()
    
    for step, batch in enumerate(train_loader):
        print(",", end="", flush=True)
        #break
        
        batch=[r.to(device) for r in batch]
        
        bert_model.zero_grad()
        
        outputs= bert_model(input_ids=batch[0], attention_mask= batch[1], labels=batch[2])
        
        loss= outputs[0]
        
        total_train_loss+=loss.item()
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)
        
        optimizer.step()
        
        scheduler.step()
        
        
        
        
    avg_train_loss = total_train_loss / len(train_loader)
    print("avg loss {} in epoch {}".format(avg_train_loss, epoch))
    train_losses.append(avg_train_loss)
    
    
    
    total_eval_loss=0
    total_eval_accuracy=0
    bert_model.eval()
    
    
    for batch in val_loader:
        print(",", end="", flush=True)
        #break        
        batch= [r.to(device) for r in batch]
        
        with torch.no_grad():
            
            outputs= bert_model(input_ids=batch[0], attention_mask= batch[1], labels=batch[2])
        
        loss= outputs[0]
        total_eval_loss+=loss.item()
        
        logits= outputs[1]
        logits = logits.detach().cpu().numpy()
        label_ids = batch[2].to('cpu').numpy()
        
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        
    avg_val_accuracy = total_eval_accuracy / len(val_loader)
    print("avg val accuracy {}".format(avg_val_accuracy))
    val_accuracy.append(avg_val_accuracy)
    
    avg_val_loss = total_eval_loss / len(val_loader)
    print("avg val loss {}".format(avg_val_loss))
    
    val_losses.append(avg_val_loss)
    
    if avg_val_loss<best_valid_loss:
        best_valid_loss=avg_val_loss
        torch.save(bert_model.state_dict(), r'C:\Users\ELECTROBOT\Desktop\model_saved_jan22.pt')
        
    
    
    print("time for epoch {} is {} minutes".format(epoch,round(time.time()-t0)/60))
    
    
    