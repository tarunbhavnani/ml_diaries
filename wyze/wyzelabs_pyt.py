# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 12:19:50 2023

@author: tarun
deep learning solution
"""

import pandas as pd
from scipy.sparse.linalg import svds
import numpy as np

train=pd.read_csv(r"C:\Users\tarun\Desktop\wyzelabs\train_rule.csv")
test= pd.read_csv(r"C:\Users\tarun\Desktop\wyzelabs\test_rule.csv")


test_trigger={}
for i in range(len(test)):
    if test.iloc[i].trigger_state not in test_trigger:
        test_trigger[test.iloc[i].trigger_state]= test.iloc[i].trigger_state_id
for i in range(len(train)):
    if train.iloc[i].trigger_state not in test_trigger:
        test_trigger[train.iloc[i].trigger_state]= train.iloc[i].trigger_state_id






test_action={}
for i in range(len(test)):
    if test.iloc[i].action not in test_action:
        test_action[str(test.iloc[i].action)]= str(test.iloc[i].action_id)
for i in range(len(train)):
    if train.iloc[i].action not in test_action:
        test_action[str(train.iloc[i].action)]= str(train.iloc[i].action_id)



# train["trig"]= [str(i)+"_"+str(j) for i,j in zip(train.trigger_device, train.trigger_state)]
# train["act"]= [str(i)+"_"+str(j) for i,j in zip(train.action, train.action_device)]

train['item']= [i+"_"+k+"_"+l+"_"+m for i,k,l,m in zip(train.trigger_device,train.trigger_state, train.action,train.action_device )]

# =============================================================================
# lets create data to train our model
# 
# =============================================================================

# kl=train.groupby("user_id")["item"].count()
# filtered_df = train.groupby("user_id").filter(lambda x: len(x) >= 2)
# data= filtered_df.groupby("user_id")["item"].apply(lambda x: list(x)).reset_index()

filtered_df1 = train.groupby(["trigger_device_id", "action_device_id"]).filter(lambda x: len(x) >= 2)
data1= filtered_df1.groupby(["trigger_device_id", "action_device_id"])["item"].apply(lambda x: list(x)).reset_index()



# Create a vocabulary mapping each word to a unique index
word_to_index = {word: idx for idx, word in enumerate(set(word for sample in data1.item for word in sample))}
index_to_word= {j:i for i,j in word_to_index.items()}



# Convert the data to indices
indexed_data = [[word_to_index[word] for word in sample] for sample in data1.item]

max_len = max(len(sample) for sample in indexed_data)
vocab_size = len(word_to_index)


new_df=[]
labels_all=[]
for num in range(len(indexed_data)):
    #break
    for nm,item in enumerate(indexed_data[num]):
        lst=indexed_data[num].copy()
        labels_all.append(lst.pop(nm))
        new_df.append(lst)


new_df=[i+[0] * (max_len - len(i)) for i in new_df]


# =============================================================================
# model
# =============================================================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(new_df, labels_all, test_size=0.33, random_state=42)


import torch

# Step 1: Define the Dataset class
class LoadDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels=None):
        self.data = torch.tensor(data, dtype=torch.long)
        #self.labels = torch.tensor(labels, dtype=torch.long)
        if labels:
            self.labels= torch.tensor(labels, dtype=torch.long)
        else:
            self.labels= None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.labels==None:
            return self.data[index]
        else:
            return self.data[index], self.labels[index]

        #return self.data[index], self.labels[index]
    

# Step 2: Create DataLoader
ld = LoadDataset(data=X_train, labels=y_train)
kl = torch.utils.data.DataLoader(ld, batch_size=10, shuffle=True)
# for i in kl:
#     break


# Step 3: Define the Model
class SportsClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super(SportsClassifier, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.fc = torch.nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)  # Aggregate along the sequence dimension
        x = self.fc(x)
        return x

# Step 4: Initialize Model, Loss Function, and Optimizer
embedding_dim = 64
num_classes = len(set(y_train))

model = SportsClassifier(vocab_size, embedding_dim, num_classes=vocab_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Step 5: Training Loop
num_epochs = 500
prev_total_train_loss = float('inf')  # Initialize with a large value
for epoch in range(num_epochs):
    total_train_loss=0
    print(epoch)
    for batch in kl:
        batch=[r.to(device) for r in batch]
        optimizer.zero_grad()
        outputs = model(batch[0])
        loss = criterion(outputs, batch[1])
        total_train_loss+=loss.item()
        loss.backward()
        optimizer.step()

    # Optionally, you can print the loss after each epoch
    avg_train_loss = total_train_loss / len(kl)
    print("avg loss {} in epoch {}".format(avg_train_loss, epoch))
    # Early exit condition
    if avg_train_loss > prev_total_train_loss:
        print("Early exit at epoch {}".format(epoch))
        break

    prev_total_train_loss = avg_train_loss



# =============================================================================
# predict
# =============================================================================


import torch.nn.functional as F
def bert_predict(model, test_dataloader):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    all_logits = []

    # For each batch in our test set...
    for batch in test_dataloader:
        batch=batch.to(device)
        # Load batch to GPU
        #b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits = model(batch)
        all_logits.append(logits)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)
    #preds = torch.argmax(all_logits, dim=1).flatten().detach().cpu().numpy()
    #top_values, top_indices = torch.topk(all_logits, k=3, dim=0)

    # Apply softmax to calculate probabilities
    #probs = F.softmax(all_logits, dim=1).cpu().numpy()

    #return list(preds)
    return all_logits


ld_test = LoadDataset(data=X_test)
kl_test = torch.utils.data.DataLoader(ld_test, batch_size=10, shuffle=False)

# Compute predicted probabilities on the test set
probs = bert_predict(model, kl_test)
probs=probs.detach().cpu().numpy()

probs=[np.argsort(i)[-4:] for i in probs]
probs=pd.DataFrame(probs)
probs["true"]=y_test


probs["acc"]=[1 if m in [i,j,k,l] else 0 for i,j,k,l,m in zip(probs[0], probs[1], probs[2],probs[3], probs["true"])]
#probs["acc"]=[1 if m in [j,k,l] else 0 for i,j,k,l,m in zip(probs[0], probs[1], probs[2],probs[3], probs["true"])]
#probs["acc"]=[1 if m in [k,l] else 0 for i,j,k,l,m in zip(probs[0], probs[1], probs[2],probs[3], probs["true"])]
#probs["acc"]=[1 if m in [l] else 0 for i,j,k,l,m in zip(probs[0], probs[1], probs[2],probs[3], probs["true"])]
acc= sum(probs["acc"])/len(probs["acc"])
#.91

# Evaluate the Bert classifier

# from sklearn.metrics import accuracy_score, f1_score
# accuracy_score(probs, y_test)
# f1_score(probs, y_test, average="weighted")
# f1_score(probs, y_test, average="micro")
# f1_score(probs, y_test, average="macro")


# =============================================================================
# lets train on fulll data and use on real test data
# =============================================================================

ld = LoadDataset(data=new_df, labels=labels_all)
kl = torch.utils.data.DataLoader(ld, batch_size=100, shuffle=True)

embedding_dim = 64
num_classes = len(set(y_train))

model = SportsClassifier(vocab_size, embedding_dim, num_classes=vocab_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Step 5: Training Loop
num_epochs = 500
prev_total_train_loss = float('inf')  # Initialize with a large value
for epoch in range(num_epochs):
    total_train_loss=0
    print(epoch)
    for batch in kl:
        batch=[r.to(device) for r in batch]
        optimizer.zero_grad()
        outputs = model(batch[0])
        loss = criterion(outputs, batch[1])
        total_train_loss+=loss.item()
        loss.backward()
        optimizer.step()

    # Optionally, you can print the loss after each epoch
    avg_train_loss = total_train_loss / len(kl)
    print("avg loss {} in epoch {}".format(avg_train_loss, epoch))
    # Early exit condition
    if avg_train_loss > prev_total_train_loss:
        print("Early exit at epoch {}".format(epoch))
        break

    prev_total_train_loss = avg_train_loss


# =============================================================================
# Prepare test data for further use 
# =============================================================================


test['item']= [i+"_"+k+"_"+l+"_"+m for i,k,l,m in zip(test.trigger_device,test.trigger_state, test.action,test.action_device )]

def predict_one(model, indexed_test):

    model.eval()

    # For each batch in our test set...
    batch= torch.tensor([indexed_test], dtype=torch.long)
    batch=batch.to(device)
    # Load batch to GPU
    #b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

    # Compute logits
    with torch.no_grad():
        logits = model(batch)
    
    probs=logits.detach().cpu().numpy()

    probs=[np.argsort(i)[-4:] for i in probs]
    
    return [i for i in probs[0]]

ids= set([i for i in test.user_id])

ss_= pd.DataFrame()

for num,user_id in enumerate(ids):
    #break
    tt= test[test.user_id==user_id]
    print(num, end=" ", flush=True)
    tt_= tt.groupby(["trigger_device_id", "action_device_id"])
    all_rules=[]
    for item in tt_:
        try:
            #break
            devices= {item[1].iloc[0]["trigger_device"]:item[1].iloc[0]["trigger_device_id"],
                      item[1].iloc[0]["action_device"]:item[1].iloc[0]["action_device_id"]}
            items= [i for i in item[1]["item"]]
            #print(items)
            indexed_test = [word_to_index[word] for word in items]
            indexed_test=indexed_test+[0] * (max_len - len(indexed_test))
            #get top three rules here
            
            rules=predict_one(model, indexed_test)
            #rules=[200,1291,278]
            rules= [index_to_word[i] for i in rules]
            
            #keep ones with trigger and action
            rules=[i for i in rules if i.split('_')[0] in devices]
            rules=[i for i in rules if i.split('_')[3] in devices]
            
            #convert to rules
            rules=[str(devices[i.split("_")[0]])+"_"+str(test_trigger[i.split("_")[1]])+"_"+str(test_action[i.split("_")[2]])+"_"+str(devices[i.split("_")[3]]) for i in rules]
            all_rules.extend(rules)
        except Exception as e:
            print(e)
        
    
    all_rules= pd.DataFrame({"rule":all_rules})
    all_rules["user_id"]=user_id
    all_rules['rank']= range(1, len(all_rules)+1)
    all_rules=all_rules[["user_id", "rule", "rank"]][:50]
    
    ss_=pd.concat([ss_,all_rules ])


    
ss_.to_csv(r"C:\Users\tarun\Desktop\wyzelabs\submisssion_16.csv", index=False)
#a78f15ff-c28f-42ac-a86f-818f26baa89a
#0.08903714213753101
    
        
        
        
        



