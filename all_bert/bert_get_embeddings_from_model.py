# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 14:47:20 2020

@author: ELECTROBOT
"""
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel


MODEL_TYPE = 'C:/Users/ELECTROBOT/Desktop/Python Projects/bert-base-uncased'
MAX_SIZE = 100
BATCH_SIZE = 50
device = torch.device('cuda'
    if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained(MODEL_TYPE)
model = BertModel.from_pretrained(MODEL_TYPE).to(device)

train_df = pd.read_csv('C:/Users/ELECTROBOT/Desktop/kaggle/twitter/train.csv')
train_df['text']=[str(i) for i in train_df['text']]

#train_df=train_df[0:10000]

test_df = pd.read_csv('C:/Users/ELECTROBOT/Desktop/kaggle/twitter/test.csv')
test_df['text']=[str(i) for i in test_df['text']]


#tokenizer.encode(train_df.text.iloc[1], add_special_tokens=True)
tokenized_input = train_df['text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
#tokenized=        train_df['text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

maxlen=max([len(i) for i in tokenized_input])

print(tokenized_input[1])
#"Here 101 -> [CLS] and 102 -> [SEP]

#padding

#padded_tokenized_input = np.array([i + [0]*(MAX_SIZE-len(i)) for i in tokenized_input.values])
padded_tokenized_input = np.array([i + [0]*(maxlen-len(i)) for i in tokenized_input])


#tell bert to ignore atteniom on padded
attention_masks  = np.where(padded_tokenized_input != 0, 1, 0)

torch.from_numpy(padded_tokenized_input[1])
#input_ids = torch.tensor(padded_tokenized_input).to(torch.int64)
#attention_masks = torch.tensor(attention_masks).to(torch.int64)

input_ids=torch.tensor([padded_tokenized_input], device=device).to(torch.int64)
attention_masks=torch.tensor([attention_masks], device=device).to(torch.int64)




all_train_embedding = []

with torch.no_grad():
  for i in tqdm(range(0,len(input_ids),2)):
    last_hidden_states = model(input_ids[i:min(i+2,len(train_df))],
                               attention_mask = attention_masks[i:min(i+2,len(train_df))])[0][:,0,:].numpy()
    all_train_embedding.append(last_hidden_states)


unbatched_train = []
for batch in all_train_embedding:
    for seq in batch:
        unbatched_train.append(seq)

#pd.DataFrame(unbatched_train).to_csv("unbatched_train.csv", index=False)

train_labels = train_df['sentiment']



# =============================================================================
# 
# =============================================================================
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt').to(device)
output = model(**encoded_input)
list(output)
output['last_hidden_state'].shape
output['pooler_output'].shape















