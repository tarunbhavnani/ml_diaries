# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 13:31:41 2023

@author: tarun
"""

import pandas as pd

train=pd.read_csv(r"C:\Users\tarun\Desktop\wyzelabs\train_rule.csv")
test= pd.read_csv(r"C:\Users\tarun\Desktop\wyzelabs\test_rule.csv")

#{i:len(set(train[i])) for i in train}
# {'user_id': 272664,
#  'trigger_device': 15,
#  'trigger_device_id': 356364,
#  'trigger_state': 45,
#  'trigger_state_id': 45,
#  'action': 47,
#  'action_id': 47,
#  'action_device': 16,
#  'action_device_id': 329798,
#  'rule': 851517}


#read


#train_d= pd.read_csv(r"C:\Users\tarun\Desktop\wyzelabs\train_device.csv")
#test_d= pd.read_csv(r"C:\Users\tarun\Desktop\wyzelabs\test_device.csv")

#ss= pd.read_csv(r"C:\Users\tarun\Desktop\wyzelabs\sample_submission.csv")
#action_map= pd.read_csv(r"C:\Users\tarun\Desktop\wyzelabs\action_map.csv")

#trigger_state_map= pd.read_csv(r"C:\Users\tarun\Desktop\wyzelabs\trigger_state_map.csv")

#trigger_state_map= pd.read_csv(r"C:\Users\tarun\Desktop\wyzelabs\trigger_state_map.csv")
#sub=pd.read_csv(r"C:\Users\tarun\Desktop\wyzelabs\sample_submission.csv")




# train.user_id.value_counts()
# test.user_id.value_counts()


# check= train[train.user_id==157343]
# check=train.copy()
# list(check)
# check=check[[ 'trigger_device', 'trigger_state', 'trigger_state_id', 'action', 'action_id', 'action_device']]
# check=check.drop_duplicates()


# kl=train.groupby([ 'trigger_device', 'trigger_state', 'trigger_state_id', 'action', 'action_id', 'action_device']).count()


# =============================================================================
# Lets find interaction for device 1 and device 2 and get the strengths!
# =============================================================================



#tr1=tr.merge(train_d, left_on="trigger_device_id", right_on="device_id", how="left")

# klp=train.groupby(["trigger_device","trigger_state","action_device", "action"]).agg({"rule":lambda x: len(x)}).reset_index()

# klp['item']= [i+"-"+k+"-"+l+"-"+m for i,k,l,m in zip(klp.trigger_device,klp.trigger_state, klp.action_device, klp.action)]

# #item_matrix
# klp=klp[["item", "rule"]]


# =============================================================================
# #can we convert the dataset in to user id against the items id and find the strngth values?
# #bases this can we preedict the nest best thing?
# 
# =============================================================================

#lets combine "trigger_device","trigger_state","action_device", "action" in one and mark against customers!

#from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import numpy as np

train['item']= [i+"_"+k+"_"+l+"_"+m for i,k,l,m in zip(train.trigger_device,train.trigger_state, train.action_device, train.action)]

mat= train[["user_id", "item"]]
op=mat.groupby(["user_id", "item"]).agg({"user_id":lambda x :len(x)})
op.columns=["strength"]
op= op.reset_index()
ui= op.pivot(index="user_id", columns="item", values="strength").fillna(0)



# matrix= ui.values
# u, s, v = svds(matrix)
# u.shape, v.shape, s.shape

# matrix_reconstructed= np.dot(np.dot(u, np.diag(s)), v)
# #normalize
# matrix_reconstructed_norm = (matrix_reconstructed - matrix_reconstructed.min()) / (matrix_reconstructed.max() - matrix_reconstructed.min())

# ui_reconstructed=pd.DataFrame(matrix_reconstructed_norm)
# ui_reconstructed=ui_reconstructed.set_index(ui.index)
# ui_reconstructed.columns= ui.columns
# see=ui_reconstructed.sample(frac=.01)

#now create a similar matrix for test using the same columns



test['item']= [i+"_"+k+"_"+l+"_"+m for i,k,l,m in zip(test.trigger_device,test.trigger_state, test.action_device, test.action)]

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
    if test.iloc[i].trigger_state_id not in test_trigger:
        test_trigger[test.iloc[i].trigger_state_id]= test.iloc[i].trigger_state

test_action={}
for i in range(len(test)):
    if test.iloc[i].action_id not in test_action:
        test_action[test.iloc[i].action_id]= test.iloc[i].action
    





#add a prefix to thye user_ids
test["user_id"]=["test_"+str(i) for i in test["user_id"]]

test_mat= test[["user_id", "item"]]


#remove all those which are not in the list of train items, check first
assert(len(test_mat)==len(test_mat[test_mat.item.isin(list(train.item))]))



#create pivot
test_op=test_mat.groupby(["user_id", "item"]).agg({"user_id":lambda x :len(x)})
test_op.columns=["strength"]
test_op= test_op.reset_index()
test_ui= test_op.pivot(index="user_id", columns="item", values="strength").fillna(0)
see=test_ui.sample(frac=.01)


final_ui=ui.append(test_ui).fillna(0)

matrix= final_ui.values
u, s, v = svds(matrix)
u.shape, v.shape, s.shape

matrix_reconstructed= np.dot(np.dot(u, np.diag(s)), v)
#normalize
matrix_reconstructed_norm = (matrix_reconstructed - matrix_reconstructed.min()) / (matrix_reconstructed.max() - matrix_reconstructed.min())

ui_reconstructed=pd.DataFrame(matrix_reconstructed_norm)
ui_reconstructed=ui_reconstructed.set_index(final_ui.index)
ui_reconstructed.columns= final_ui.columns


# #lets calculate for test_4
# new_user_latent_factors = u[ui_reconstructed.index.get_loc("test_4")]
# #new_user_latent_factors=np.array(ui_reconstructed.loc["test_4"].values).reshape(-1)
# predicted_ratings = np.dot(new_user_latent_factors, v)


# # Get top-N recommended items
# #num_recommendations = 1000
# recommended_item_indices = np.argsort(predicted_ratings)[::-1]#[:num_recommendations]
# recommended_items = final_ui.columns[recommended_item_indices]

# listed_devices= test[test.user_id=="test_4"]
# listed_devices=list(listed_devices.trigger_device)+list(listed_devices.action_device)

# recommended_items=recommended_items[[True if i.split("-")[0] in listed_devices and i.split("-")[2] in listed_devices  else False for i in recommended_items]]
# recommended_item=recommended_items[0]

# print("Recommended items for new user:")
# print(recommended_items)


#lets predict for all


# =============================================================================
# ids= [i for i in test.user_id]
# 
# ss_= pd.DataFrame(columns= ["user_id", "rule", "rank"])
# 
# #user_id= "test_19"
# 
# for user_id in ids:
#     print(".", end=" ", flush=True)
#     temp=pd.DataFrame()
#     #break
#     new_user_latent_factors = u[ui_reconstructed.index.get_loc(user_id)]
#     #new_user_latent_factors=np.array(ui_reconstructed.loc["test_4"].values).reshape(-1)
#     predicted_ratings = np.dot(new_user_latent_factors, v)
#     # Get top-N recommended items
#     #num_recommendations = 1000
#     recommended_item_indices = np.argsort(predicted_ratings)[::-1]#[:num_recommendations]
#     recommended_items = final_ui.columns[recommended_item_indices]
#     #check which ones are applicable to the uesr_id
#     
#     listed_devices= test[test.user_id==user_id]
#     #listed_devices.trigger_device
#     listed_devices=list(set(list(listed_devices.trigger_device)+list(listed_devices.action_device)))
#     
#     recommended_items=recommended_items[[True if i.split("_")[0] in listed_devices and i.split("_")[2] in listed_devices  else False for i in recommended_items]]
#     recommended_item=pd.DataFrame(recommended_items[0:50]).reset_index()
#     recommended_item.columns= ["rank", "rule"]
#     recommended_item["user_id"]=user_id
#     
#     ss_=ss_.append(recommended_item)
#     
#     
# #one issue we are predictibg somethin glike "Camera-Detects a person-Camera-Upload a short video to the cloud" but many cameras for eg
# #ro care of this we might have to do something more here
# 
# 
# #now we have to update the device names with the device ids
# 
# #lets do that for test_19
# #follow above code till recommended_item
# #recommended_item
# recommended_item[['trigger_device', 'trigger', 'action_device', 'action']] = recommended_item['rule'].str.split('_', expand=True)
# 
# recommended_item["trigger_device_id"]=[[i for i,j in test_device["test_19"].items() if j==k] for k in recommended_item["trigger_device"]]
# recommended_item["action_device_id"]=[[i for i,j in test_device["test_19"].items() if j==k] for k in recommended_item["action_device"]]
# 
# recommended_item["trigger_id"]=recommended_item["trigger"].map({j:i for i,j in test_trigger.items()})
# recommended_item["action_id"]=recommended_item["action"].map({j:i for i,j in test_action.items()})
# 
# #now get all the combinations of the trigger_device id and action_device ids
# 
# 
# all_rules=[]
# for i in range(len(recommended_item)):
#     #break
#     #i=0
#     
#     trigger_device=recommended_item.iloc[i]['trigger_device_id'] if isinstance(recommended_item.iloc[i]['trigger_device_id'], list) else [recommended_item.iloc[i]['trigger_device_id']]
#     trigger = recommended_item.iloc[i]['trigger_id'] if isinstance(recommended_item.iloc[i]['trigger_id'], list) else [recommended_item.iloc[i]['trigger_id']]
#     
#     #action_device=recommended_item.iloc[i].action_device_id
#     action_device = recommended_item.iloc[i]['action_device_id'] if isinstance(recommended_item.iloc[i]['action_device_id'], list) else [recommended_item.iloc[i]['action_device_id']]
#     #action=recommended_item.iloc[i].action_id
#     action = recommended_item.iloc[i]['action_id'] if isinstance(recommended_item.iloc[i]['action_id'], list) else [recommended_item.iloc[i]['action_id']]
#     multi_index = pd.MultiIndex.from_product([trigger_device, trigger, action_device, action])
#     
#     rules= [str(i[0])+"_"+str(i[1])+"_"+str(i[2])+"_"+str(i[3]) for i in multi_index]
#     all_rules+=rules
# 
# 
# =============================================================================

# =============================================================================
# now lets get this for all the test ids
# =============================================================================


ids= [i for i in test.user_id]

ss_= pd.DataFrame(columns= ["user_id", "rule", "rank"])

#user_id= "test_19"

for user_id in ids:
    print(user_id, end=" ", flush=True)
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
    
    recommended_items=recommended_items[[True if i.split("_")[0] in listed_devices and i.split("_")[2] in listed_devices  else False for i in recommended_items]]
    recommended_item=pd.DataFrame(recommended_items[0:50]).reset_index()
    recommended_item.columns= ["rank", "rule"]
    recommended_item["user_id"]=user_id
    
    recommended_item[['trigger_device', 'trigger', 'action_device', 'action']] = recommended_item['rule'].str.split('_', expand=True)
    recommended_item["trigger_device_id"]=[[i for i,j in test_device[user_id].items() if j==k] for k in recommended_item["trigger_device"]]
    recommended_item["action_device_id"]=[[i for i,j in test_device[uesr_id].items() if j==k] for k in recommended_item["action_device"]]
    recommended_item["trigger_id"]=recommended_item["trigger"].map({j:i for i,j in test_trigger.items()})
    recommended_item["action_id"]=recommended_item["action"].map({j:i for i,j in test_action.items()})

    #now get all the combinations of the trigger_device id and action_device ids


    all_rules=[]
    for i in range(len(recommended_item)):
        #break
        #i=0
        
        trigger_device=recommended_item.iloc[i]['trigger_device_id'] if isinstance(recommended_item.iloc[i]['trigger_device_id'], list) else [recommended_item.iloc[i]['trigger_device_id']]
        trigger = recommended_item.iloc[i]['trigger_id'] if isinstance(recommended_item.iloc[i]['trigger_id'], list) else [recommended_item.iloc[i]['trigger_id']]
        
        #action_device=recommended_item.iloc[i].action_device_id
        action_device = recommended_item.iloc[i]['action_device_id'] if isinstance(recommended_item.iloc[i]['action_device_id'], list) else [recommended_item.iloc[i]['action_device_id']]
        #action=recommended_item.iloc[i].action_id
        action = recommended_item.iloc[i]['action_id'] if isinstance(recommended_item.iloc[i]['action_id'], list) else [recommended_item.iloc[i]['action_id']]
        multi_index = pd.MultiIndex.from_product([trigger_device, trigger, action_device, action])
        
        rules= [str(i[0])+"_"+str(i[1])+"_"+str(i[2])+"_"+str(i[3]) for i in multi_index]
        all_rules+=rules
    all_rules= all_rules[:50]
    
    temp=pd.DataFrame({"rule":all_rules})
    temp["user_id"]=user_id
    temp["rank"]=[i for i in range(len(temp))]
    ss_= ss_.append(temp)
    
ss__= ss_.copy()
ss_["user_id"]= [int(i.split("_")[1]) for i in ss_.user_id]
ss_["rank"]=[i+1 for i in ss_["rank"]]


ss_["rule"]=["_".join([i.split("_")[0],i.split("_")[1],i.split("_")[3],i.split("_")[2]]) for i in ss_["rule"]]


ss_.to_csv(r"C:\Users\tarun\Desktop\wyzelabs\submisssion.csv", index=False)


ss_10= ss_.groupby(['user_id']).head(10)
ss_10.to_csv(r"C:\Users\tarun\Desktop\wyzelabs\sample_submission.csv", index=False)








