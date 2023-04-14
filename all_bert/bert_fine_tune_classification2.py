# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 20:16:41 2020

@author: ELECTROBOT
"""

import os
os.chdir('C:\\Users\\ELECTROBOT\\Desktop\\Python Projects\\Bert-model')
import pandas as pd
import numpy as np


data = pd.read_csv("./ner_data_kaggle/ner_dataset.csv", encoding="latin1").fillna(method="ffill")
data=data[0:1000]
data.head(10)

class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


getter = SentenceGetter(data)

sentences=[[j[0] for j in i] for i in getter.sentences]
sentences[0]
labels=[[j[2] for j in i] for i in getter.sentences]

tag_values = list(set(data['Tag']))
tag_values.append("PAD")
tag2idx = {t: i for i, t in enumerate(tag_values)}


#lets apply bert
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler
from transformers import BertTokenizer, BertConfig
from sklearn.model_selection import train_test_split


maxlen= 75
bs= 32


device = torch.device("cuda")
n_gpu=torch.cuda.device_count()


torch.cuda.get_device_name(0)



#tokenizer= BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case= False)
#tokenizer =BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
import tokenizers
tokenizer = tokenizers.BertWordPieceTokenizer(r'C:\Users\ELECTROBOT\Desktop\flask_bert_ml\bert_sentiment_flask\input\bert_base_uncased/vocab.txt',lowercase=True)
#can not directly use the bert tokenizer since the tokens are not mere words but broken words.
#hence we will create it specially now



def tkn(sent, tag):
    sentence_tokens=[]
    label_tokens=[]

    for s, t in zip(sent,tag):
        #s1= tokenizer.tokenize(s)
        s1= tokenizer.encode(s)
        sentence_tokens.extend(s1.ids[1:-1])
        label_tokens.extend([t]*len(s1.ids[1:-1]))
    return sentence_tokens, label_tokens

klo= [tkn(i,j) for i,j in zip(sentences, labels)]    


tokenizer.decode([2610,2056,1996,   2329,   2188,   2436,   4912,   2019,   4812,   1997,   1047,   26115,   2213,   1005,   1055,   5248])
#"police said the british home office sought an investigation of khayam's behavior"



input_ids=[i for i,j in klo]

#input_ids=[tokenizer.convert_tokens_to_ids(i) for i in input_ids]
labels=[[tag2idx[j] for j in i] for i in labels]


#trip and pad
min([len(i) for i in input_ids])

input_ids=[i+[0]*(maxlen-len(i)) if len(i)<75 else i for i in input_ids]#pad
input_ids=[i[0:75] for i in input_ids]#trim


labels=[i+[17]*(maxlen-len(i)) if len(i)<75 else i for i in labels]#pad
labels=[i[0:75] for i in labels]#trim


#we also need to create masks , can create from any of the two. its 0 for pads 1s otherwise

input_ids=np.array(input_ids)
attention_masks= np.where(input_ids==0,0,1)


input_ids= torch.tensor(input_ids).to(torch.int64)
labels= torch.tensor(labels).to(torch.int64)
attention_masks= torch.tensor(attention_masks).to(torch.int64)



#data done

tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, labels,
                                                            random_state=2018, test_size=0.1)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=0.1)






train_data= TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler= RandomSampler(train_data)
train_dataloader= DataLoader(train_data, sampler=train_sampler, batch_size=bs)


valid_data= TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler= SequentialSampler(valid_data)
valid_dataloader= DataLoader(valid_data, sampler= valid_sampler, batch_size=bs)



import transformers

model=transformers.BertForTokenClassification.from_pretrained('bert-base-cased',
                                              num_labels= len(tag2idx),
                                              output_attentions=False,
                                              output_hidden_states=False)


model.save_pretrained(r"C:\Users\ELECTROBOT\Desktop\bert-classify")



model.cuda()



param_optimizer= list(model.classifier.named_parameters())
optimizer_grouped_parameters= [{"params":[p for n,p in param_optimizer]}]
optimizer = transformers.AdamW(
    optimizer_grouped_parameters,
    lr=3e-5,
    eps=1e-8
)

from transformers import get_linear_schedule_with_warmup

epochs = 3
max_grad_norm = 1.0

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)


from seqeval.metrics import f1_score, accuracy_score


loss_values, validation_loss_values = [], []

for _ in range(0, epochs):
    
    model.train()
    total_loss=0
    
    
    for step, batch in enumerate(train_dataloader):
        batch= tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels= batch
        model.zero_grad()
        outputs= model(b_input_ids,token_type_ids=None,
                       attention_mask= b_input_mask,
                       labels= b_labels)
        loss= outputs[0]
        loss.backward()
        total_loss+=loss.item()
        if step % 100==99:
            print(total_loss, loss.item())

        #print(total_loss, loss.item())
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        scheduler.step()
    avg_train_loss= total_loss/len(train_dataloader)
    print("Average train loss: {}".format(avg_train_loss))
    loss_values.append(avg_train_loss)
    
    #add validation code below
    
    #validation
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    
    for step, batch in enumerate(valid_dataloader):
        #print(1)
        
        batch= tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels= batch
        
        with torch.no_grad():
            outputs= model(b_input_ids, token_type_ids=None,
                           attention_mask= b_input_mask,
                           labels= b_labels)
        logits= outputs[1].cpu().numpy()
        label_ids= b_labels.to('cpu').numpy()
            
        # Calculate the accuracy for this batch of test sentences.
        eval_loss += outputs[0].mean().item()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)
        if step % 100==99:
            print(eval_loss,outputs[0].mean().item())
            
        
    eval_loss = eval_loss / len(valid_dataloader)
    validation_loss_values.append(eval_loss)
    print("Validation loss: {}".format(eval_loss))
    pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                 for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
    valid_tags = [tag_values[l_i] for l in true_labels
                                  for l_i in l if tag_values[l_i] != "PAD"]
    print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
    print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
    print()
            
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
    
    











