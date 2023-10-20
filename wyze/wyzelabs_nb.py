# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 19:27:11 2023

@author: tarun

break in trg and action, trigs are users actions are items work with items ui matrix
"""

#lets redo everythingnand consider trgger dev and trggger togetehr same for action


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





# =============================================================================

train["trig"]= [str(i)+"_"+str(j) for i,j in zip(train.trigger_device, train.trigger_state)]
train["act"]= [str(i)+"_"+str(j) for i,j in zip(train.action, train.action_device)]

train['rule']= [i+j for i,j in zip(train.trig, train.act)]


kl= train.groupby('user_id')['rule'].apply(lambda x: list(x)).reset_index()

kl=kl[[True if len(i)>1 else False for i in kl.rule]]

data= [i for i in kl.rule]

vocab= {}
counter=0
for io in data:
    for i in io:
        if i not in vocab:
            vocab[i]=counter
            counter+=1
            

data_= [[vocab[j] for j in i] for i in data]



# Hyperparameters
vocab_size = len(vocab)  # Size of your vocabulary
embedding_dim = 16
hidden_dim = 32
num_layers = 2
batch_size = 1

# Initialize the model, loss function, and optimizer
model = LanguageModel(vocab_size, embedding_dim, hidden_dim, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create a DataLoader for training
train_dataset = CustomDataset(data_)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(5):
    print(epoch)
    total_loss = 0
    for batch in train_loader:
        print(".", end="", flush=True)

        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output.view(-1, vocab_size), batch.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss}")

# Optionally, save the model
torch.save(model.state_dict(), 'language_model.pth')
import os
os.getcwd()
os.chdir(r"C:\Users\tarun\Desktop\check")


# Define a function for word prediction
def predict_next_word(model, input_seq, vocab_size):
    model.eval()
    with torch.no_grad():
        input_seq = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0)
        output = model(input_seq)
        _, predicted_idx = torch.max(output[:, -1, :], 1)
        return predicted_idx.item()

# Load the trained model (assuming it's saved as 'language_model.pth')
model = LanguageModel(vocab_size, embedding_dim, hidden_dim, num_layers)
model.load_state_dict(torch.load('language_model.pth'))
model.eval()

# Define your vocabulary (replace with your own)
#idx_to_word = {0: 'word1', 1: 'word2', 2: 'word3', 3: 'word4', 4: 'word5'}
#word_to_idx = {v: k for k, v in idx_to_word.items()}

# Example usage
#data_[5]
partial_sequence = [18, 19, 20, 21, 22, 23, 24, 25,18]
predicted_idx = predict_next_word(model, partial_sequence, vocab_size)

print(f"The predicted next word is: {predicted_word}")

#BS, it predicts the last term present in the sequence

# =============================================================================


kl=train[1:2000].groupby("trig")["act"].apply(lambda x: list(x)).reset_index()
#lets say trig is user and act is item

#create ui matrix

mat= train[["trig", "act"]]
mat["strength"]=1
op=mat.groupby(["trig", "act"])["strength"].count().reset_index()

ui= op.pivot(index="trig", columns="act", values="strength").fillna(0)





# =============================================================================
# 
# =============================================================================



kl=train.groupby(["trigger_device_id","action_device_id"])["act"].apply(lambda x: list(x)).reset_index()









