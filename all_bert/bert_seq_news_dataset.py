# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 12:29:03 2021

@author: ELECTROBOT
"""

# =============================================================================
# get data
# =============================================================================
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import re

dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data
target= dataset.target

df= pd.DataFrame({'sentence':documents, 'label':target})

df['sentence']= [re.sub(r'[^A-Za-z /.]'," ",i) for i in df['sentence']]
df['sentence']= [re.sub(r'\s+'," ",i) for i in df['sentence']]


# Get the lists of sentences and their labels.
sentences = df.sentence.values
labels = df.label.values



#helper accuracy function
import numpy as np

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# =============================================================================
# load bert
# =============================================================================


from transformers import BertTokenizer
bert_path=r"C:\Users\ELECTROBOT\Desktop\bert_base_uncased"

# Load the BERT tokenizer.

tokenizer = BertTokenizer.from_pretrained(bert_path, do_lower_case=True)





# =============================================================================
# create data loader
# =============================================================================



import torch
class loader(torch.utils.data.Dataset):
    def __init__(self, tokenizer,documents, labels=None, max_len=512):
        super().__init__()
        self.documents= documents
        if labels:
            self.labels=torch.tensor(labels)
        else:
            self.labels=None
        self.encoded= tokenizer.batch_encode_plus(self.documents, padding="max_length", max_length= max_len, truncation=True, return_attention_mask=True)
        
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



#load= loader(tokenizer=tokenizer, documents=sentences, labels= labels.tolist())


# kl=torch.utils.data.DataLoader(load, batch_size=4, shuffle=True)
# for i in kl:
#     break

# =============================================================================
# create train and val dataloader
# =============================================================================

from sklearn.model_selection import train_test_split

train,val, y_train, y_val= train_test_split(sentences, labels, test_size=.3, stratify= labels, shuffle=True)



train_loader= loader(tokenizer=tokenizer, documents=train, labels= y_train.tolist())
train_loader= torch.utils.data.DataLoader(train_loader, batch_size=4, shuffle=True)


val_loader= loader(tokenizer=tokenizer, documents=val, labels= y_val.tolist())
val_loader= torch.utils.data.DataLoader(val_loader, batch_size=4, shuffle=False)




# =============================================================================
# Define model
# =============================================================================


from transformers import BertForSequenceClassification, AdamW, BertConfig
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs=5

model = BertForSequenceClassification.from_pretrained(bert_path, num_labels=len(set(target)),
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

model.to(device)

params = list(model.named_parameters())


optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )


#set warm-up
from transformers import get_linear_schedule_with_warmup
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)




# =============================================================================
# train model
# =============================================================================


import time
t0=time.time()

best_valid_loss = float('inf')
train_losses=[]
val_losses=[]
val_accuracy=[]



for epoch in range(0,epochs):
    
    print("epoch---{}".format(epoch))
    
    total_train_loss=0
    
    model.train()
    
    for step, batch in enumerate(train_loader):
        print(",", end="", flush=True)
        #break
        
        batch=[r.to(device) for r in batch]
        
        model.zero_grad()
        
        outputs= model(input_ids=batch[0], attention_mask= batch[1], labels=batch[2])
        
        loss= outputs[0]
        
        total_train_loss+=loss.item()
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        scheduler.step()
        
        
        
        
    avg_train_loss = total_train_loss / len(train_loader)
    print("avg loss {} in epoch {}".format(avg_train_loss, epoch))
    train_losses.append(avg_train_loss)
    
    
    
    total_eval_loss=0
    total_eval_accuracy=0
    model.eval()
    
    
    for batch in val_loader:
        print(",", end="", flush=True)
        #break        
        batch= [r.to(device) for r in batch]
        
        with torch.no_grad():
            
            outputs= model(input_ids=batch[0], attention_mask= batch[1], labels=batch[2])
        
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
        torch.save(model.state_dict(), r'C:\Users\ELECTROBOT\Desktop\saved_model\model_saved.pt')
        
    
    
    print("time for epoch {} is {} minutes".format(epoch,round(time.time()-t0)/60))
    
    
    


#can train the model for a couple of epochs on the val data as well!!
        
# =============================================================================
# save full model and tokenizer
# =============================================================================
        
model.save_pretrained(r'C:\Users\ELECTROBOT\Desktop\saved_model\full')
tokenizer.save_pretrained(r'C:\Users\ELECTROBOT\Desktop\saved_model\full')
 
            
            
            
# =============================================================================
# predict on new data, load keys in bert
# =============================================================================

from transformers import BertForSequenceClassification,BertTokenizer
import torch
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bert_path=r"C:\Users\ELECTROBOT\Desktop\bert_base_uncased"
model = BertForSequenceClassification.from_pretrained(bert_path, num_labels=20,
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

model.to(device)


path=r'C:\Users\ELECTROBOT\Desktop\saved_model\model_saved.pt'
model.load_state_dict(torch.load(path))

tokenizer = BertTokenizer.from_pretrained(bert_path, do_lower_case=True)



doc= 'Although I realize that principle is not one of your strongest points I would still like to know why do do not ask any question of this sort about the Arab countries. If you want to continue this think tank charade of yours your fixation on Israel must stop. You might have to start asking the same sort of questions of Arab countries as well. You realize it would not work as the Arab countries treatment of Jews over the last several decades is so bad that your fixation on Israel would begin to look like the biased attack that it is. Everyone in this group recognizes that your stupid Center for Policy Research is nothing more than a fancy name for some bigot who hates Israel.'

#17

encoded= tokenizer.encode_plus(doc, padding="max_length", max_length= 512, truncation=True, return_attention_mask=True)




ids=torch.tensor(encoded['input_ids']).to(device)

#encoded['token_type_ids']
masks=torch.tensor(encoded['attention_mask']).to(device)

with torch.no_grad():
        
        # model predictions
        preds = model(input_ids=ids.unsqueeze(0), attention_mask= masks.unsqueeze(0))
        
pred=np.argmax(preds['logits'].detach().cpu().numpy(), axis=1)
        
# =============================================================================
# load full saved model        
# =============================================================================
    
    


from transformers import BertForSequenceClassification,BertTokenizer
import torch
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mdl= BertForSequenceClassification.from_pretrained(r'C:\Users\ELECTROBOT\Desktop\saved_model\full')
mdl.to(device)
tok= BertTokenizer.from_pretrained(r'C:\Users\ELECTROBOT\Desktop\saved_model\full', do_lower_case=True)




doc= 'Although I realize that principle is not one of your strongest points I would still like to know why do do not ask any question of this sort about the Arab countries. If you want to continue this think tank charade of yours your fixation on Israel must stop. You might have to start asking the same sort of questions of Arab countries as well. You realize it would not work as the Arab countries treatment of Jews over the last several decades is so bad that your fixation on Israel would begin to look like the biased attack that it is. Everyone in this group recognizes that your stupid Center for Policy Research is nothing more than a fancy name for some bigot who hates Israel.'

#17

encoded= tok.encode_plus(doc, padding="max_length", max_length= 512, truncation=True, return_attention_mask=True)

ids=torch.tensor(encoded['input_ids']).to(device)

#encoded['token_type_ids']
masks=torch.tensor(encoded['attention_mask']).to(device)

with torch.no_grad():
        # model predictions
        preds = mdl(input_ids=ids.unsqueeze(0), attention_mask= masks.unsqueeze(0))
        
pred=np.argmax(preds['logits'].detach().cpu().numpy(), axis=1)



# =============================================================================
# end
# =============================================================================

# =============================================================================
# predict on val dataset
# =============================================================================

#for a list of inputs, not using the loader
predicted=[]

for va in val:
    print(".", end="", flush=True)
    encoded= tokenizer.encode_plus(va, padding="max_length", max_length= 512, truncation=True, return_attention_mask=True)
    ids=torch.tensor(encoded['input_ids']).to(device)

    #encoded['token_type_ids']
    masks=torch.tensor(encoded['attention_mask']).to(device)
    
    with torch.no_grad():
            
            # model predictions
            preds = model(input_ids=ids.unsqueeze(0), attention_mask= masks.unsqueeze(0))
            
    pred=np.argmax(preds['logits'].detach().cpu().numpy(), axis=1)
    predicted.append(pred)
    
from sklearn import metrics

cr= metrics.classification_report(y_val, predicted)

#               precision    recall  f1-score   support

#            0       0.67      0.56      0.61       144
#            1       0.73      0.70      0.72       175
#            2       0.75      0.71      0.73       177
#            3       0.65      0.72      0.68       177
#            4       0.79      0.70      0.74       173
#            5       0.83      0.85      0.84       178
#            6       0.80      0.83      0.82       176
#            7       0.52      0.76      0.62       178
#            8       0.82      0.72      0.77       180
#            9       0.91      0.86      0.88       179
#           10       0.92      0.89      0.90       180
#           11       0.89      0.82      0.85       179
#           12       0.74      0.70      0.72       177
#           13       0.86      0.85      0.86       178
#           14       0.86      0.81      0.84       178
#           15       0.75      0.66      0.70       180
#           16       0.78      0.74      0.76       164
#           17       0.82      0.82      0.82       169
#           18       0.73      0.75      0.74       140
#           19       0.30      0.42      0.35       113

#     accuracy                           0.75      3395
#    macro avg       0.76      0.74      0.75      3395
# weighted avg       0.77      0.75      0.76      3395











