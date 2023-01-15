# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 18:26:14 2022

@author: ELECTROBOT
"""
import os
import torch
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, train_test_split

df= pd.DataFrame({"abc": np.random.choice(np.arange(1000), 10000, replace=True), "xyz":np.random.choice(np.arange(1000), 10000, replace=True)})



X_train, X_test, y_train, y_test = train_test_split( train, train['target'], test_size=0.2, random_state=4, shuffle=True)

# =============================================================================
# make k folds
# =============================================================================

config= {"this":"that"}

def make_folds(df: pd.DataFrame,  config:type, cv_schema=None) -> pd.DataFrame:
    """Split the given dataframe into training folds."""
    #TODO: add options for cv_scheme.
    df_folds = df.copy()
    skf = StratifiedKFold(5, shuffle=True, random_state=121)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X=df_folds['abc'], y=df_folds['xyz'])):
        df_folds.loc[val_idx, 'fold'] = int(fold+1)
    df_folds['fold'] = df_folds['fold'].astype(int)
    print(df_folds.groupby(['fold', 'xyz']).size())

    return df_folds

make_folds(df, config)


# =============================================================================
# Make seed 
# =============================================================================
def seed_all(seed: int = 1930):

    print("Using Seed Number {}".format(seed))

    os.environ["PYTHONHASHSEED"] = str(
        seed)  # set PYTHONHASHSEED env var at fixed value
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)  # pytorch (both CPU and CUDA)
    np.random.seed(seed)  # for numpy pseudo-random generator
    random.seed(
        seed)  # set fixed value for python built-in pseudo-random generator
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
seed_all(seed=123)



# =============================================================================
# data loader
# =============================================================================

#load tokenized data
class loader(torch.utils.data.Dataset):
    def __init__(self, tokenized_ds):
        self.data = tokenized_ds

    def __getitem__(self, index):
        item = {k: self.data[k][index] for k in self.data.keys()}
        return item

    def __len__(self):
        return len(self.data['input_ids'])
    


#load data direct

class loader(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, labels=None,max_len=512):
        self.data=data
        
        if labels:
            self.labels=torch.tensor(labels)
        else:
            self.labels=None
        
        self.encoded= tokenizer.batch_encode_plus(self.documents,
                                                  padding="max_length",
                                                  max_length=max_len,
                                                  truncation=True,
                                                  return_attention_mask=True)    
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = {k: self.encoded[k][index] for k in self.encoded.keys()}
        if self.labels!=None:
            labels= self.labels[index]
            return item, labels
        else:
            return item



load= loader(**)

kl=torch.utils.data.DataLoader(load, batch_size=4, shuffle=True)

        
    


# =============================================================================
# Tokenizers
# =============================================================================

from transformers import BertTokenizer
model_path=r"C:\Users\ELECTROBOT\Desktop\model_dump\bert_base_uncased"
tokenizer= BertTokenizer.from_pretrained(model_path)

#or call using AutoTokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)

tokenizer.encode("this is a tokenizer")
# [101, 2023, 2003, 1037, 19204, 17629, 102]

tokenizer.encode_plus("this is a tokenizer")
#{'input_ids': [101, 2023, 2003, 1037, 19204, 17629, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1]}

tokenizer.encode_plus(["this is a tokenizer", "this is batch one"])
#bert#{'input_ids': [101, 100, 100, 102], 'token_type_ids': [0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1]}
# this one behaves differently when tokenizer loaded with bert and autotokenizer.
#auto#{'input_ids': [101, 2023, 2003, 1037, 19204, 17629, 102, 2023, 2003, 14108, 2028, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

tokenizer.batch_encode_plus(["this is a tokenizer", "this is batch one"])
#{'input_ids': [[101, 2023, 2003, 1037, 19204, 17629, 102], [101, 2023, 2003, 14108, 2028, 102]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]}


#roberta
model_path_r= r'C:\Users\ELECTROBOT\Desktop\model_dump\roberta'
tokenizer = AutoTokenizer.from_pretrained(model_path_r, add_prefix_space=True)

# =============================================================================
# tokenizer = AutoTokenizer.from_pretrained(model_path_r, add_prefix_space=False)
# 
# 
# tokenizer.encode(" this is a tokenizer")
# #[0, 42, 16, 10, 19233, 6315, 2]
# tokenizer.encode("this is a tokenizer")
# #[0, 9226, 16, 10, 19233, 6315, 2]
# 
# #see the difference this. add_prefix_space (bool, optional, defaults to False) — Whether or not to add an initial space to the input. 
# #This allows to treat the leading word just as any other word. (RoBERTa tokenizer detect beginning of words by the preceding space).
# #When used with is_split_into_words=True, this tokenizer needs to be instantiated with add_prefix_space=True.
# 
# =============================================================================
tokenizer.encode_plus("this is a tokenizer")
#{'input_ids': [0, 42, 16, 10, 19233, 6315, 2], 'attention_mask': [1, 1, 1, 1, 1, 1, 1]}

tokenizer.encode_plus(["this is a tokenizer", "this is batch one"])
#{'input_ids': [0, 42, 16, 10, 19233, 6315, 2, 2, 42, 16, 14398, 65, 2], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}


tokenizer.batch_encode_plus(["this is a tokenizer", "this is batch one"])




# =============================================================================
# trainer
# =============================================================================










# =============================================================================
# optimizers and schedulers
# =============================================================================





# =============================================================================
# accuracy
# =============================================================================











# =============================================================================
# inference 
# =============================================================================
