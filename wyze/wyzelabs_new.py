# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 19:27:11 2023

@author: tarun

break in trg and action, trigs are users actions are items work with items ii matrix
"""

#lets redo everythingnand consider trgger dev and trggger togetehr same for action


import pandas as pd

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

#decomposing matrix is used when we have to predsict new movies or new items which the user might not have seen/used.
#but here we have to predict the probable ones not anything new,*


#if i create an item item matrix and then reconstruct it
#now i predict for the user, just remove all which are not probable and select the most probables?


#lets see!

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



# #user_user

# uu=pd.DataFrame(cosine_similarity(ui,ui))


# # Create a MinMaxScaler
# scaler = MinMaxScaler()

# # Fit and transform the similarity matrix
# uu_normalized = scaler.fit_transform(uu)

# # Convert the result back to a DataFrame
# uu_normalized = pd.DataFrame(uu_normalized, columns=ui.index, index=ui.index)



#user_id=19

#tt= test[test.user_id==user_id]

#items= {i[0]+"_"+i[1]: str(i[2])+"_"+str(i[3]) for i in zip(tt.trigger_device, tt.trigger_state, tt.trigger_device_id, tt.trigger_state_id)}


#we will go through each item and check which is the most probable next action!
#we will store these actions and remove which have happened, remaining should be th emost probable ones

ids= set([i for i in test.user_id])

ss_= pd.DataFrame()

for num,user_id in enumerate(ids):
    tt= test[test.user_id==user_id]
    print(num, end=" ", flush=True)
    all_rules=pd.DataFrame()
    for i in range(len(tt)):
        
        
        trigger= tt.trigger_device.iloc[i]+"_"+ tt.trigger_state.iloc[i]
        action=  tt.action.iloc[i]+"_"+ tt.action_device.iloc[i]
        
        all_probable_actions= ii_normalized.loc[action].reset_index()
        all_probable_actions.columns= ["act", "score"]
        #all_probable_actions=all_probable_actions.sort_values(by="score", ascending=False)
        #all_probable_actions['rank_a']=range(1,len(all_probable_actions)+1)
        
        #all possible actions as per trigger
        all_probable_actions_on_trigger=ui.loc[trigger].reset_index()
        all_probable_actions_on_trigger.columns= ["act", "score"]
        std=scaler.fit_transform(all_probable_actions_on_trigger[["score"]])
        all_probable_actions_on_trigger["score"]=[i[0] for i in std]
    
        
        #all_probable_actions_on_trigger=all_probable_actions_on_trigger.sort_values(by="score", ascending=False)
        #all_probable_actions_on_trigger['rank_at']=range(1,len(all_probable_actions_on_trigger)+1)
        
        #all_probable_actions_on_trigger=all_probable_actions_on_trigger[all_probable_actions_on_trigger.score>0]
        all_probable_actions_on_trigger=all_probable_actions_on_trigger.merge(all_probable_actions, on="act", how="left")
        all_probable_actions_on_trigger["score"]=all_probable_actions_on_trigger["score_x"]+all_probable_actions_on_trigger["score_y"]
        
        #all_probable_actions_on_trigger["rank"]=all_probable_actions_on_trigger["rank_a"]+all_probable_actions_on_trigger["rank_at"]
        all_probable_actions_on_trigger=all_probable_actions_on_trigger.sort_values(by="score", ascending=False)
        
        
        #remove all which are not the same as the action device
        all_probable_actions_on_trigger=all_probable_actions_on_trigger[[True if j.split("_")[1]==tt.iloc[i].action_device else False for j in all_probable_actions_on_trigger["act"] ]]
        
        
        #covert rules to '1426277_25_41_263477' format
        trigger_rule= str(tt.trigger_device_id.iloc[i])+"_"+str(tt.trigger_state_id.iloc[i])
        
        all_probable_actions_on_trigger["rule"]=[trigger_rule+"_"+str(test_action[j.split("_")[0]])+"_"+str(tt.iloc[i].action_device_id) for j in all_probable_actions_on_trigger["act"]]
        
        # rules= all_probable_actions_on_trigger["act"][0:50]
        # rules=[j for j in rules if j.split("_")[1]==tt.iloc[i].action_device]
        
        
        # #covert rules to '1426277_25_41_263477' format
        # trigger_rule= str(tt.trigger_device_id.iloc[i])+"_"+str(tt.trigger_state_id.iloc[i])
        # rules= [trigger_rule+"_"+str(test_action[j.split("_")[0]])+"_"+str(tt.iloc[i].action_device_id) for j in rules]
        
        # #remove already present, take top 3
        # rules=[i for i in rules if i not in list(tt.rule)][:10]
        all_rules=pd.concat([all_rules,all_probable_actions_on_trigger ])
        
    
    all_rules= all_rules.sort_values(by="score", ascending=False)
    all_rules["user_id"]=user_id
    all_rules=all_rules[[False if i in list(tt.rule) else True for i in all_rules.rule]]
    all_rules=all_rules.drop_duplicates(subset='rule')
    
    all_rules["rank"]=range(1, len(all_rules)+1)
    
    all_rules=all_rules[["user_id", "rule", "rank"]][:50]
    
    ss_=pd.concat([ss_,all_rules ])


    
ss_.to_csv(r"C:\Users\tarun\Desktop\wyzelabs\submisssion_10.csv", index=False)
    
#ef2819fc-6766-4734-8911-b5c660fcfcb2
#.1241, didnt remove the already ones and didnt drop duplicates

#f6022770-34bb-4550-ade2-24017aeb68f1
#.24 best till now
