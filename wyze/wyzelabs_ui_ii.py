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


#item-item
from sklearn.metrics.pairwise import cosine_similarity

ii=pd.DataFrame(cosine_similarity(ui.T,ui.T))

from sklearn.preprocessing import MinMaxScaler

# Create a MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the similarity matrix
ii_normalized = scaler.fit_transform(ii)

# Convert the result back to a DataFrame
ii_normalized = pd.DataFrame(ii_normalized, columns=ui.columns, index=ui.columns)



#now lets go through test and preeict for each one and add

#user_id=65538
ids= set([i for i in test.user_id])

ss_= pd.DataFrame()

for num,user_id in enumerate(ids):

    tt= test[test.user_id==user_id]
    print(num, end=" ", flush=True)
    
    trigger_action= list(set([i for i in zip(tt.trigger_device_id, tt.action_device_id,tt.trigger_device,tt.action_device)]))
    
    all_rules=[]
    for i,j,k,l in trigger_action:
        #break
        subset= tt[(tt.trigger_device_id==i) & (tt.action_device_id==j)]
        fdf=[]
        for trigger_ in set(subset.trigger_state):
            subset_=subset[subset.trigger_state==trigger_]
            trigger= subset_.trigger_device.iloc[0]+"_"+ subset_.trigger_state.iloc[0]
            
            
            #using ui
            all_probable_actions=ui_reconstructed.loc[trigger].reset_index()
            all_probable_actions.columns=["act", "score"]
            all_probable_actions=all_probable_actions[[True if i.split('_')[1]==l else False for i in all_probable_actions["act"]]]
            all_probable_actions["act"]=[test_action[i.split("_")[0]]+"_"+str(j) for i in all_probable_actions['act']]
            all_probable_actions["act"]=[str(i)+"_"+str(test_trigger[trigger_])+"_"+act for act in all_probable_actions["act"]]
            fdf.append(all_probable_actions)
            
            
            #using ii
            dfs = []
            for action_ in set(subset.action):
                #break
                action=  action_+"_"+ subset_.action_device.iloc[0]
            
                all_probable_actions_= ii_normalized.loc[action].reset_index()
                all_probable_actions_.columns= ["act", "score"]
                all_probable_actions_=all_probable_actions_[[True if i.split('_')[1]==l else False for i in all_probable_actions_["act"]]]
                all_probable_actions_["act"]=[test_action[i.split("_")[0]]+"_"+str(j) for i in all_probable_actions_['act']]
                dfs.append(all_probable_actions_)
            
            final_df = pd.concat(dfs, ignore_index=True)
            final_df=final_df.groupby("act")["score"].apply(lambda x: np.mean(x)).reset_index()
            #final_df=final_df[~final_df.act.isin([i+"_"+subset_.action_device.iloc[0] for i in set(subset.action)])]
            
            final_df["act"]=[str(i)+"_"+str(test_trigger[trigger_])+"_"+act for act in final_df["act"]]
            fdf.append(final_df)
            
            
        fdf = pd.concat(fdf, ignore_index=True)
        fdf=fdf.groupby("act")["score"].sum().reset_index()
        all_rules.append(fdf)
    all_rules = pd.concat(all_rules, ignore_index=True)
    all_rules=all_rules.groupby("act")["score"].sum().reset_index()
    all_rules=all_rules.sort_values(by="score", ascending=False)
    all_rules=all_rules[[False if i in list(tt.rule) else True for i in all_rules['act']]][0:50]
    all_rules.columns=["rule","score"]
    all_rules["rank"]=range(1, len(all_rules)+1)
    all_rules["user_id"]=user_id
    
    all_rules=all_rules[["user_id", "rule", "rank"]]
 
    ss_=pd.concat([ss_,all_rules ])
    
ss_.to_csv(r"C:\Users\tarun\Desktop\wyzelabs\submisssion_12.csv", index=False)
#82d7e130-9e17-461f-acf3-18ccbbfbaf99
#0.23820040416053365

#1426277_25_41_


