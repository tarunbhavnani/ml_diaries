# -*- coding: utf-8 -*-
"""
Created on Fri May  7 17:04:18 2021

@author: ELECTROBOT
"""

# =============================================================================
# get data
# =============================================================================
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import re
from sklearn.model_selection import train_test_split
dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data
target= dataset.target

df= pd.DataFrame({'sentence':documents, 'label':target})
#df,_,_,_=train_test_split(df,df['label'], test_size=.7, stratify= df['label'], random_state=3)

df['sentence']= [re.sub(r'[^A-Za-z /.]'," ",i) for i in df['sentence']]
df['sentence']= [re.sub(r'\s+'," ",i) for i in df['sentence']]


# Get the lists of sentences and their labels.
sentences = df.sentence.values
labels = df.label.values

# =============================================================================
# 
# =============================================================================



train_texts, val_texts, train_labels, val_labels = train_test_split(sentences, labels, test_size=.2, stratify=labels)

from transformers import BertTokenizerFast
bert_path=r"C:\Users\ELECTROBOT\Desktop\bert_base_uncased"
tokenizer = BertTokenizerFast.from_pretrained(r"C:\Users\ELECTROBOT\Desktop\bert_base_uncased")

max_len=200

train_encodings=tokenizer.batch_encode_plus(
    train_texts.tolist(),
    max_length = max_len,
    pad_to_max_length=True,
    truncation=True
)


val_encodings = tokenizer.batch_encode_plus(val_texts.tolist(),max_length = max_len,
    padding=True,
    truncation=True)

# test_encodings = tokenizer.batch_encode_plus(test_texts.tolist(),max_length = max_len,
#     padding=True,
#     truncation=True)





import torch

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)
#test_dataset = IMDbDataset(test_encodings, test_labels)



# =============================================================================
# 
# =============================================================================


from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AdamW

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#device = torch.device('cpu')

model = BertForSequenceClassification.from_pretrained(
    bert_path, # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = len(set(labels)))
 

#model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)#gpu
#train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)#cpu

optim = AdamW(model.parameters(), lr=5e-5)
all_losses=[]
for epoch in range(10):
    print(epoch)
    for batch in train_loader:
        print(".", end="", flush=True)
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].long().to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        all_losses.append(loss)
        loss.backward()
        optim.step()

torch.save(model.state_dict(), 'saved_weights_hf_full_df_10epochs.pt')


i=0
losses_np=[]
for i in range(0,len(all_losses),10):
    batch_loss= all_losses[i:i+10]
    losses_np.append(sum([i.detach().cpu().numpy() for i in batch_loss])/10)

import matplotlib.pyplot as plt
plt.plot(losses_np)



import numpy as np

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)



model.eval()
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

total_eval_accuracy = 0
total_eval_loss = 0

for batch in val_loader:
    print(".", end="", flush=True)
    input_ids= batch['input_ids'].to(device)
    attention_mask= batch['attention_mask'].to(device)
    labels= batch['labels'].long().to(device)
    with torch.no_grad():
        outputs= model(input_ids, attention_mask=attention_mask, labels=labels)
    total_eval_loss+=outputs[0].detach().cpu().numpy()
    logits= outputs[1].detach().cpu().numpy()
    label_ids= labels.to('cpu').numpy()
    total_eval_accuracy += flat_accuracy(logits, label_ids)
    
        

avg_val_accuracy = total_eval_accuracy / len(val_loader)


#0.7180583501006036# 2 epochs
#0.7349094567404427# 10epcohs





