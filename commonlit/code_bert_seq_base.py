# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 12:00:56 2021

@author: ELECTROBOT
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split
import spacy
nlp= spacy.load('en_core_web_sm')
os.chdir(r'C:\Users\ELECTROBOT\Desktop\kaggle\commonlit_readbility')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train= pd.read_csv('train_june13.csv')

train['target']=[i+i*j for i,j in zip(train.target, train.standard_error)]

train['avg_sent_len']=[(i-min(train['avg_sent_len']))/(max(train['avg_sent_len'])-min(train['avg_sent_len'])) for i in train['avg_sent_len']]
train['dale_words']=[(i-min(train['dale_words']))/(max(train['dale_words'])-min(train['dale_words'])) for i in train['dale_words']]
train['hard_word']=[(i-min(train['hard_word']))/(max(train['hard_word'])-min(train['hard_word'])) for i in train['hard_word']]
train['spacy_trial']=[(i-min(train['spacy_trial']))/(max(train['spacy_trial'])-min(train['spacy_trial'])) for i in train['spacy_trial']]

from transformers import AdamW
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AutoConfig
from transformers import get_cosine_schedule_with_warmup
from transformers import BertTokenizer, BertModel, BertConfig,BertForSequenceClassification, AdamW
path=r"C:\Users\ELECTROBOT\Desktop\bert_base_uncased"
tokenizer= BertTokenizer.from_pretrained(path)


MAX_LEN=256
#data loader
class loader(torch.utils.data.Dataset):
    def __init__(self, df, test=True):
        super().__init__()

        self.df = df        
        self.test = test
        self.text = df.excerpt.tolist()
        self.avg_sent_len= df.avg_sent_len.tolist()
        self.dale_words= df.dale_words.tolist()
        self.hard_word= df.hard_word.tolist()
        self.spacy_trial= df.spacy_trial.tolist()
        #self.text = [text.replace("\n", " ") for text in self.text]
        
        if not self.test:
            self.target = torch.tensor(df.target.values, dtype=torch.float32)
    
        self.encoded = tokenizer.batch_encode_plus(
            self.text,
            padding = 'max_length',            
            max_length = MAX_LEN,
            truncation = True,
            return_attention_mask=True
        )        
 

    def __len__(self):
        return len(self.df)

    
    def __getitem__(self, index):        
        input_ids = torch.tensor(self.encoded['input_ids'][index])
        attention_mask = torch.tensor(self.encoded['attention_mask'][index])
        avg_sent_len=torch.tensor(self.avg_sent_len[index])
        dale_words=torch.tensor(self.dale_words[index])
        hard_word=torch.tensor(self.hard_word[index])
        spacy_trial=torch.tensor(self.spacy_trial[index])
        
        if self.test:
            return (input_ids, attention_mask,avg_sent_len,dale_words,hard_word,spacy_trial)            
        else:
            target = self.target[index]
            return (input_ids, attention_mask, avg_sent_len,dale_words,hard_word,spacy_trial,target)




model = BertForSequenceClassification.from_pretrained(path, num_labels=1)



# =============================================================================
# train function
# =============================================================================

def train_model():
  
  model.train()

  total_loss, total_accuracy = 0, 0
  
  # empty list to save model predictions
  total_preds=[]
  
  # iterate over batches
  for step, batch in enumerate(train_loader):
    if step % 50 == 0 and not step == 0:
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_loader)))
      
    # push the batch to gpu
    batch = [r.to(device) for r in batch]
    
    input_ids, attention_mask, avg_sent_len,dale_words,hard_word,spacy_trial,target = batch
    
    # clear previously calculated gradients 
    model.zero_grad()
    
    # get model predictions for the current batch
    outputs = model(input_ids, attention_mask=attention_mask, labels=target)
    
    # compute the loss between actual and predicted values
    
    loss=outputs[0]
    
    
    # add on to the total loss
    total_loss = total_loss + loss.item()
    
    # backward pass to calculate the gradients
    loss.backward()
    
    # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # update parameters
    optimizer.step()
    
    # model predictions are stored on GPU. So, push it to CPU
    
    
    # append the model predictions
    
    
  # compute the training loss of the epoch
  avg_loss = total_loss / len(train_loader)
  
  # predictions are in the form of (no. of batches, size of batch, no. of classes).
  # reshape the predictions in form of (number of samples, no. of classes)
  

  #returns the loss and predictions
  return avg_loss, total_preds


# =============================================================================
# evaluate function
# =============================================================================
def evaluate():
  
  print("\nEvaluating...")
  
  # deactivate dropout layers
  model.eval()

  total_loss = 0
  
  # empty list to save the model predictions
  total_preds = []

  # iterate over batches
  for step, batch in enumerate(val_loader):
      #print(step)
      #break
    
    # Progress update every 50 batches.
    if step % 50 == 0 and not step == 0:
      
      # Calculate elapsed time in minutes.
      #elapsed = format_time(time.time() - t0)
            
      # Report progress.
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_loader)))

    # push the batch to gpu
    batch = [t.to(device) for t in batch]

    input_ids, attention_mask, avg_sent_len,dale_words,hard_word,spacy_trial,target = batch

    # deactivate autograd
    with torch.no_grad():
      
      # model predictions
      outputs = model(input_ids, attention_mask=attention_mask, labels=target)
      
      loss=outputs[0]
      # compute the validation loss between actual and predicted values
      #loss = cross_entropy(preds,labels)

      total_loss = total_loss + loss.item()

      

      

  # compute the validation loss of the epoch
  avg_loss = total_loss / len(val_loader) 

  # reshape the predictions in form of (number of samples, no. of classes)
  

  return avg_loss, total_preds






# =============================================================================
# call model and optimizer
# =============================================================================

from transformers import AdamW
import time
from sklearn.model_selection import KFold
# # define the optimizer
# optimizer = AdamW(model.parameters(),lr = .0001)    

#optimizer = torch.optim.SGD(model.parameters(), lr =.0001 )




# set initial loss to infinite
best_valid_loss = float('inf')

# empty lists to store training and validation loss of each epoch
train_losses=[]
valid_losses=[]
epochs=10
models_saved=[]


kfold = KFold(n_splits=5, random_state=10, shuffle=True)

start = time.time()
for fold, (train_indices, val_indices) in enumerate(kfold.split(train)): 
    
    print(f"\nFold {fold + 1}/{5}")
    #break
    train_df= train.loc[train_indices]
    dataset=loader(train_df, test=False)
    train_loader=torch.utils.data.DataLoader(dataset, batch_size=4,drop_last=True, shuffle=True)

    val_df= train.loc[val_indices]
    dataset=loader(val_df, test=False)
    val_loader=torch.utils.data.DataLoader(dataset, batch_size=4,drop_last=True, shuffle=False)
    
    model = BertForSequenceClassification.from_pretrained(path, num_labels=1)
    model.to(device)
    optimizer = AdamW(model.parameters(),lr = .0001)  
    
    
    for epoch in range(epochs):
     
        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
    
        
        train_loss, _ = train_model()
    
        
        valid_loss, _ = evaluate()
        
        #save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            #torch.save(model.state_dict(), 'saved_weights_full_model.pt')
            torch.save(model.state_dict(), 'final_model/model_saved_bs.pt')
            models_saved.append((fold,epoch))
        
        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        
        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')
        
end = time.time()   

(end-start)/60 #42 mins



# =============================================================================
# load best model
# =============================================================================



#load weights of best model
path = 'final_model/model_saved_bs.pt'
model.load_state_dict(torch.load(path))

# =============================================================================
# predict
# =============================================================================

#get valdiatoin df from fold where best model was saved#

dataset=loader(val_df, test=False)
test_loader=torch.utils.data.DataLoader(dataset, batch_size=1,drop_last=True, shuffle=False)


total_preds=[]
for step,batch in enumerate(test_loader):
    print(step)
    batch = [t.to(device) for t in batch]
    input_ids, attention_mask, avg_sent_len,dale_words,hard_word,spacy_trial,target = batch
    #labels = batch['labels'].to(device).reshape(attention_mask.shape[0],-1)
    with torch.no_grad():
    
        outputs = model(input_ids, attention_mask=attention_mask)        
   
        preds = outputs['logits'].detach().cpu().numpy()
    
        total_preds.append(preds)

#preds= predict()


tt=pd.DataFrame({'text': val_df.excerpt, 'actual':val_df['target'], 'preds': total_preds})
val_df['pred']=preds
