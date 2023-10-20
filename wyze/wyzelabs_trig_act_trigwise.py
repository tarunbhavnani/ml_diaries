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


#lets say trig is user and act is item

#create ui matrix

mat= train[["trig", "act"]]
mat["strength"]=1
op=mat.groupby(["trig", "act"])["strength"].count().reset_index()

ui= op.pivot(index="trig", columns="act", values="strength").fillna(0)

matrix= ui.values
u, s, v = svds(matrix)
u.shape, v.shape, s.shape

matrix_reconstructed= np.dot(np.dot(u, np.diag(s)), v)
#normalize
matrix_reconstructed_norm = (matrix_reconstructed - matrix_reconstructed.min()) / (matrix_reconstructed.max() - matrix_reconstructed.min())

ui_reconstructed=pd.DataFrame(matrix_reconstructed_norm)
ui_reconstructed=ui_reconstructed.set_index(ui.index)
ui_reconstructed.columns= ui.columns



#now lets go through test and preeict for each one and add

#user_id=19

ids= set([i for i in test.user_id])

ss_= pd.DataFrame()

for num,user_id in enumerate(ids):
    tt= test[test.user_id==user_id]
    print(num, end=" ", flush=True)

    all_rules=pd.DataFrame()
    for i in range(len(tt)):
        
        
        trigger= tt.trigger_device.iloc[i]+"_"+ tt.trigger_state.iloc[i]
        
        all_probable_actions=ui_reconstructed.loc[trigger].reset_index()
        all_probable_actions.columns=["act", "score"]
    
        #remove all which are not the same as the action device
        all_probable_actions=all_probable_actions[[True if j.split("_")[1]==tt.iloc[i].action_device else False for j in all_probable_actions["act"] ]]
        
        
        #remove the current action
        #all_probable_actions
        
    
    
        #covert rules to '1426277_25_41_263477' format
        trigger_rule= str(tt.trigger_device_id.iloc[i])+"_"+str(tt.trigger_state_id.iloc[i])
        
        all_probable_actions["rule"]=[trigger_rule+"_"+str(test_action[j.split("_")[0]])+"_"+str(tt.iloc[i].action_device_id) for j in all_probable_actions["act"]]
    
        all_rules=pd.concat([all_rules,all_probable_actions ])
        
    
    
    all_rules= all_rules.sort_values(by="score", ascending=False)
    all_rules["user_id"]=user_id
    all_rules=all_rules[[False if i in list(tt.rule) else True for i in all_rules.rule]]
    all_rules=all_rules.drop_duplicates(subset='rule')
    
    all_rules["rank"]=range(1, len(all_rules)+1)
    
    all_rules=all_rules[["user_id", "rule", "rank"]][:50]
    
    ss_=pd.concat([ss_,all_rules ])


    
ss_.to_csv(r"C:\Users\tarun\Desktop\wyzelabs\submisssion_11.csv", index=False)
(381480, 3))

#1433f264-93bf-4e7c-bbd4-6a872c1ba550
#.2534 best 


#1426277_25_41_


