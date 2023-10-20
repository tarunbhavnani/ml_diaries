# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 13:31:41 2023

@author: tarun
"""

import pandas as pd

train=pd.read_csv(r"C:\Users\tarun\Desktop\wyzelabs\train_rule.csv")
test= pd.read_csv(r"C:\Users\tarun\Desktop\wyzelabs\test_rule.csv")




# =============================================================================
# Lets find interaction for device 1 and device 2 and get the strengths!
# =============================================================================



#from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

train['item']= [i+"_"+k+"_"+l+"_"+m for i,k,l,m in zip(train.trigger_device,train.trigger_state, train.action,train.action_device )]

mat= train[["user_id", "item"]]
op=mat.groupby(["user_id", "item"]).agg({"user_id":lambda x :len(x)})
op.columns=["strength"]
op= op.reset_index()
ui= op.pivot(index="user_id", columns="item", values="strength").fillna(0)




test['item']= [i+"_"+k+"_"+l+"_"+m for i,k,l,m in zip(test.trigger_device,test.trigger_state,  test.action,test.action_device)]

test_device={}
for j in list(set(test.user_id)):
    tt= test[test.user_id==j]
    td={}
    for i in range(len(tt)):
        if tt.iloc[i].trigger_device_id not in td:
            td[tt.iloc[i].trigger_device_id]= tt.iloc[i].trigger_device
        if tt.iloc[i].action_device_id not in td:
            td[tt.iloc[i].action_device_id]= tt.iloc[i].action_device
    
    test_device[j]=td



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





#add a prefix to thye user_ids

#test["user_id"]=["test_"+str(i) for i in test["user_id"]]

test_mat= test[["user_id", "item"]]


#remove all those which are not in the list of train items, check first
assert(len(test_mat)==len(test_mat[test_mat.item.isin(list(train.item))]))



#create pivot
test_op=test_mat.groupby(["user_id", "item"]).agg({"user_id":lambda x :len(x)})
test_op.columns=["strength"]
test_op= test_op.reset_index()
test_ui= test_op.pivot(index="user_id", columns="item", values="strength").fillna(0)
#hd=test_ui.sample(frac=.01)


final_ui=ui.append(test_ui).fillna(0)

from scipy.sparse.linalg import svds
matrix= final_ui.values
u, s, v = svds(matrix)
u.shape, v.shape, s.shape

matrix_reconstructed= np.dot(np.dot(u, np.diag(s)), v)
#normalize
matrix_reconstructed_norm = (matrix_reconstructed - matrix_reconstructed.min()) / (matrix_reconstructed.max() - matrix_reconstructed.min())

ui_reconstructed=pd.DataFrame(matrix_reconstructed_norm)
ui_reconstructed=ui_reconstructed.set_index(final_ui.index)
ui_reconstructed.columns= final_ui.columns



#lets predict for all


# =============================================================================
# now lets get this for all the test ids
# =============================================================================


ids= set([i for i in test.user_id])

ss_= pd.DataFrame(columns= ["user_id", "rule", "rank"])

#user_id= 19
#tt= test[test.user_id==user_id]


for num,user_id in enumerate(ids):
    tt= test[test.user_id==user_id]
    print(num, end=" ", flush=True)
    temp=pd.DataFrame()
    #break
    new_user_latent_factors = u[ui_reconstructed.index.get_loc(user_id)]
    #new_user_latent_factors=np.array(ui_reconstructed.loc["test_4"].values).reshape(-1)
    predicted_ratings = np.dot(new_user_latent_factors, v)
    # Get top-N recommended items
    #num_recommendations = 1000
    recommended_item_indices = np.argsort(predicted_ratings)[::-1]#[:num_recommendations]
    recommended_items = final_ui.columns[recommended_item_indices]
    #check which ones are applicable to the uesr_id
    
    listed_devices= test[test.user_id==user_id]
    #listed_devices.trigger_device
    listed_devices=list(set(list(listed_devices.trigger_device)+list(listed_devices.action_device)))
    
    recommended_items=recommended_items[[True if i.split("_")[0] in listed_devices and i.split("_")[3] in listed_devices  else False for i in recommended_items]]
    recommended_item=pd.DataFrame(recommended_items[0:50]).reset_index()
    recommended_item.columns= ["rank", "rule"]
    recommended_item["user_id"]=user_id
    recommended_item=recommended_item[["user_id", "rule", "rank"]]
    
    
    recommended_item[['trigger_device', 'trigger', 'action', 'action_device']] = recommended_item['rule'].str.split('_', expand=True)
    
    #lets tag the devices
    all_rules=[]
    for i in range(len(recommended_item)):
        #if both same!
        if recommended_item.iloc[i]["trigger_device"]==recommended_item.iloc[i]["action_device"]:
            device=recommended_item.iloc[i]["trigger_device"]
            device_ids=[i for i,j in test_device[user_id].items() if j==device]
            rule=[str(j)+"_"+str(test_trigger[recommended_item.iloc[i]["trigger"]])+"_"+str(test_action[recommended_item.iloc[i]["action"]])+"_"+str(j) for j in device_ids]
            all_rules+=rule
        else:
            
            #find groups of these devices in real data
            devices_both=[recommended_item.iloc[i]["trigger_device"],recommended_item.iloc[i]["action_device"]]
            device_ids=[(j,l) for i,j,k,l in zip(tt['trigger_device'],tt['trigger_device_id'],tt['action_device'],tt['action_device_id']) if i==devices_both[0] and k==devices_both[1]]
            rule=[str(j[0])+"_"+str(test_trigger[recommended_item.iloc[i]["trigger"]])+"_"+str(test_action[recommended_item.iloc[i]["action"]])+"_"+str(j[1]) for j in device_ids]
            all_rules+=rule
    
    all_rules = [item for index, item in enumerate(all_rules) if item not in all_rules[:index]]

    all_rules=[i for i in all_rules if i not in list(tt.rule)]
    
    all_rules= all_rules[:50]
    
    temp=pd.DataFrame({"rule":all_rules})
    temp["user_id"]=user_id
    temp["rank"]=[i for i in range(1,len(temp)+1)]
    temp=temp[["user_id", "rule", "rank"]]
    ss_= ss_.append(temp)
    

#ss_['rank']=[i+1 for i in ss_['rank']]
ss_.to_csv(r"C:\Users\tarun\Desktop\wyzelabs\submisssion_7.csv", index=False)
#70917c6a-176c-4a28-80e3-50337c16e0d7
#.17 best till now






