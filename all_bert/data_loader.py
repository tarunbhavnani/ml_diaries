# -*- coding: utf-8 -*-
"""
Created on Tue May 11 09:37:37 2021

@author: ELECTROBOT
"""
import pandas as pd
training_data= pd.read_csv(r"C:\Users\ELECTROBOT\Desktop\kaggle\datasets\imdb\IMDB Dataset.csv")


from torch.utils.data import DataLoader

#train_loader=DataLoader(training_data, batch_size=64, shuffle=True)


tr_dt= [(i,j) for i,j in zip(training_data['review'],training_data['sentiment'])]

train_loader=DataLoader(tr_dt, batch_size=64, shuffle=True)

hj, jk=next(iter(train_loader))