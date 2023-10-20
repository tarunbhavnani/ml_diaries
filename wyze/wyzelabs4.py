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
    if test.iloc[i].trigger_state_id not in test_trigger:
        test_trigger[test.iloc[i].trigger_state_id]= test.iloc[i].trigger_state

test_action={}
for i in range(len(test)):
    if test.iloc[i].action_id not in test_action:
        test_action[test.iloc[i].action_id]= test.iloc[i].action
    





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
hd=test_ui.sample(frac=.01)


final_ui=ui.append(test_ui).fillna(0)

from scipy.sparse.linalg import svds
import numpy as np
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


for user_id in ids:
    tt= test[test.user_id==user_id]
    print(user_id, end=" ", flush=True)
    temp=pd.DataFrame()
    #break
    new_user_latent_factors = u[ui_reconstructed.index.get_loc(user_id)]
    #new_user_latent_factors=np.array(ui_reconstructed.loc["test_4"].values).reshape(-1)
    predicted_ratings = np.dot(new_user_latent_factors, v)
    
    #top_n = 10
    #recommended_item_indices = np.argsort(predicted_ratings)[-top_n:][::-1]
    recommended_item_indices = np.argsort(predicted_ratings)[::-1]
    recommended_items = ui.columns[recommended_item_indices]
    # Get the predicted ratings for the top N items
    top_n_predicted_ratings = predicted_ratings[recommended_item_indices]

    preds={}
    for i,j in zip(recommended_items,top_n_predicted_ratings):
        if i not in preds:
            preds[i]=j
        else:
            preds[i]+=j
    preds = dict(sorted(preds.items(), key=lambda item: item[1],reverse=True))
    #preds=dict(list(preds.items())[:10])
    preds={i:j for i,j in preds.items() if i not in list(tt["item"])}


    top_quartile_cutoff = np.percentile([i for i in preds.values()], 75)

    preds={i:j for i,j in preds.items() if j >=top_quartile_cutoff}
    recommended_items=[i for i in preds]
    # Get top-N recommended items
    #num_recommendations = 1000
    #recommended_item_indices = np.argsort(predicted_ratings)[::-1]#[:num_recommendations]
    #recommended_items = final_ui.columns[recommended_item_indices]
    #check which ones are applicable to the uesr_id
    
    listed_devices= test[test.user_id==user_id]
    #listed_devices.trigger_device
    listed_devices=list(set(list(listed_devices.trigger_device)+list(listed_devices.action_device)))
    
    #recommended_items=recommended_items[[True if i.split("_")[0] in listed_devices and i.split("_")[3] in listed_devices  else False for i in recommended_items]]
    recommended_items=[i for i in recommended_items if i.split("_")[0] in listed_devices and i.split("_")[3] in listed_devices]
    
    if recommended_items!=[]:
        #recommended_item=pd.DataFrame(recommended_items[0:50]).reset_index()
        recommended_item=pd.DataFrame(recommended_items).reset_index()
        
        recommended_item.columns= ["rank", "rule"]
        recommended_item["user_id"]=user_id
        recommended_item=recommended_item[["user_id", "rule", "rank"]]
        
        
        recommended_item[['trigger_device', 'trigger', 'action', 'action_device']] = recommended_item['rule'].str.split('_', expand=True)
        
        recommended_item["trigger_device_id"]=[[i for i,j in test_device[user_id].items() if j==k] for k in recommended_item["trigger_device"]]
        
        
        recommended_item["action_device_id"]=[[i for i,j in test_device[user_id].items() if j==k] for k in recommended_item["action_device"]]
        
        
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
            multi_index = pd.MultiIndex.from_product([trigger_device, trigger, action,action_device ])
            
            rules= [str(i[0])+"_"+str(i[1])+"_"+str(i[2])+"_"+str(i[3]) for i in multi_index]
            all_rules+=rules
        all_rules= all_rules[:50]
        
        temp=pd.DataFrame({"rule":all_rules})
        temp["user_id"]=user_id
        temp["rank"]=[i for i in range(1,len(temp)+1)]
        temp=temp[["user_id", "rule", "rank"]]
        ss_= ss_.append(temp)
    

#ss_['rank']=[i+1 for i in ss_['rank']]
ss_.to_csv(r"C:\Users\tarun\Desktop\wyzelabs\submisssion_3.csv", index=False)





#sub_10=  ss_.groupby('user_id').head(10)
#sub_10.to_csv(r"C:\Users\tarun\Desktop\wyzelabs\sub_10.csv", index=False)


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
    if test.iloc[i].trigger_state_id not in test_trigger:
        test_trigger[test.iloc[i].trigger_state_id]= test.iloc[i].trigger_state

test_action={}
for i in range(len(test)):
    if test.iloc[i].action_id not in test_action:
        test_action[test.iloc[i].action_id]= test.iloc[i].action
    





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
hd=test_ui.sample(frac=.01)


final_ui=ui.append(test_ui).fillna(0)

from scipy.sparse.linalg import svds
import numpy as np
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


for user_id in ids:
    tt= test[test.user_id==user_id]
    print(user_id, end=" ", flush=True)
    temp=pd.DataFrame()
    #break
    new_user_latent_factors = u[ui_reconstructed.index.get_loc(user_id)]
    #new_user_latent_factors=np.array(ui_reconstructed.loc["test_4"].values).reshape(-1)
    predicted_ratings = np.dot(new_user_latent_factors, v)
    
    #top_n = 10
    #recommended_item_indices = np.argsort(predicted_ratings)[-top_n:][::-1]
    recommended_item_indices = np.argsort(predicted_ratings)[::-1]
    recommended_items = ui.columns[recommended_item_indices]
    # Get the predicted ratings for the top N items
    top_n_predicted_ratings = predicted_ratings[recommended_item_indices]

    preds={}
    for i,j in zip(recommended_items,top_n_predicted_ratings):
        if i not in preds:
            preds[i]=j
        else:
            preds[i]+=j
    preds = dict(sorted(preds.items(), key=lambda item: item[1],reverse=True))
    #preds=dict(list(preds.items())[:10])
    preds={i:j for i,j in preds.items() if i not in list(tt["item"])}


    top_quartile_cutoff = np.percentile([i for i in preds.values()], 75)

    preds={i:j for i,j in preds.items() if j >=top_quartile_cutoff}
    recommended_items=[i for i in preds]
    # Get top-N recommended items
    #num_recommendations = 1000
    #recommended_item_indices = np.argsort(predicted_ratings)[::-1]#[:num_recommendations]
    #recommended_items = final_ui.columns[recommended_item_indices]
    #check which ones are applicable to the uesr_id
    
    listed_devices= test[test.user_id==user_id]
    #listed_devices.trigger_device
    listed_devices=list(set(list(listed_devices.trigger_device)+list(listed_devices.action_device)))
    
    #recommended_items=recommended_items[[True if i.split("_")[0] in listed_devices and i.split("_")[3] in listed_devices  else False for i in recommended_items]]
    recommended_items=[i for i in recommended_items if i.split("_")[0] in listed_devices and i.split("_")[3] in listed_devices]
    
    if recommended_items!=[]:
        #recommended_item=pd.DataFrame(recommended_items[0:50]).reset_index()
        recommended_item=pd.DataFrame(recommended_items).reset_index()
        
        recommended_item.columns= ["rank", "rule"]
        recommended_item["user_id"]=user_id
        recommended_item=recommended_item[["user_id", "rule", "rank"]]
        
        
        recommended_item[['trigger_device', 'trigger', 'action', 'action_device']] = recommended_item['rule'].str.split('_', expand=True)
        
        recommended_item["trigger_device_id"]=[[i for i,j in test_device[user_id].items() if j==k] for k in recommended_item["trigger_device"]]
        
        
        recommended_item["action_device_id"]=[[i for i,j in test_device[user_id].items() if j==k] for k in recommended_item["action_device"]]
        
        
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
            multi_index = pd.MultiIndex.from_product([trigger_device, trigger, action,action_device ])
            
            rules= [str(i[0])+"_"+str(i[1])+"_"+str(i[2])+"_"+str(i[3]) for i in multi_index]
            all_rules+=rules
        all_rules= all_rules[:50]
        
        temp=pd.DataFrame({"rule":all_rules})
        temp["user_id"]=user_id
        temp["rank"]=[i for i in range(1,len(temp)+1)]
        temp=temp[["user_id", "rule", "rank"]]
        ss_= ss_.append(temp)
    

#ss_['rank']=[i+1 for i in ss_['rank']]
ss_.to_csv(r"C:\Users\tarun\Desktop\wyzelabs\submisssion_3.csv", index=False)





#sub_10=  ss_.groupby('user_id').head(10)
#sub_10.to_csv(r"C:\Users\tarun\Desktop\wyzelabs\sub_10.csv", index=False)


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
    if test.iloc[i].trigger_state_id not in test_trigger:
        test_trigger[test.iloc[i].trigger_state_id]= test.iloc[i].trigger_state

test_action={}
for i in range(len(test)):
    if test.iloc[i].action_id not in test_action:
        test_action[test.iloc[i].action_id]= test.iloc[i].action
    





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
hd=test_ui.sample(frac=.01)


final_ui=ui.append(test_ui).fillna(0)

from scipy.sparse.linalg import svds
import numpy as np
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


for user_id in ids:
    tt= test[test.user_id==user_id]
    print(user_id, end=" ", flush=True)
    temp=pd.DataFrame()
    #break
    new_user_latent_factors = u[ui_reconstructed.index.get_loc(user_id)]
    #new_user_latent_factors=np.array(ui_reconstructed.loc["test_4"].values).reshape(-1)
    predicted_ratings = np.dot(new_user_latent_factors, v)
    
    #top_n = 10
    #recommended_item_indices = np.argsort(predicted_ratings)[-top_n:][::-1]
    recommended_item_indices = np.argsort(predicted_ratings)[::-1]
    recommended_items = ui.columns[recommended_item_indices]
    # Get the predicted ratings for the top N items
    top_n_predicted_ratings = predicted_ratings[recommended_item_indices]

    preds={}
    for i,j in zip(recommended_items,top_n_predicted_ratings):
        if i not in preds:
            preds[i]=j
        else:
            preds[i]+=j
    preds = dict(sorted(preds.items(), key=lambda item: item[1],reverse=True))
    #preds=dict(list(preds.items())[:10])
    preds={i:j for i,j in preds.items() if i not in list(tt["item"])}


    top_quartile_cutoff = np.percentile([i for i in preds.values()], 75)

    preds={i:j for i,j in preds.items() if j >=top_quartile_cutoff}
    recommended_items=[i for i in preds]
    # Get top-N recommended items
    #num_recommendations = 1000
    #recommended_item_indices = np.argsort(predicted_ratings)[::-1]#[:num_recommendations]
    #recommended_items = final_ui.columns[recommended_item_indices]
    #check which ones are applicable to the uesr_id
    
    listed_devices= test[test.user_id==user_id]
    #listed_devices.trigger_device
    listed_devices=list(set(list(listed_devices.trigger_device)+list(listed_devices.action_device)))
    
    #recommended_items=recommended_items[[True if i.split("_")[0] in listed_devices and i.split("_")[3] in listed_devices  else False for i in recommended_items]]
    recommended_items=[i for i in recommended_items if i.split("_")[0] in listed_devices and i.split("_")[3] in listed_devices]
    
    if recommended_items!=[]:
        #recommended_item=pd.DataFrame(recommended_items[0:50]).reset_index()
        recommended_item=pd.DataFrame(recommended_items).reset_index()
        
        recommended_item.columns= ["rank", "rule"]
        recommended_item["user_id"]=user_id
        recommended_item=recommended_item[["user_id", "rule", "rank"]]
        
        
        recommended_item[['trigger_device', 'trigger', 'action', 'action_device']] = recommended_item['rule'].str.split('_', expand=True)
        
        recommended_item["trigger_device_id"]=[[i for i,j in test_device[user_id].items() if j==k] for k in recommended_item["trigger_device"]]
        
        
        recommended_item["action_device_id"]=[[i for i,j in test_device[user_id].items() if j==k] for k in recommended_item["action_device"]]
        
        
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
            multi_index = pd.MultiIndex.from_product([trigger_device, trigger, action,action_device ])
            
            rules= [str(i[0])+"_"+str(i[1])+"_"+str(i[2])+"_"+str(i[3]) for i in multi_index]
            all_rules+=rules
        all_rules= all_rules[:50]
        
        temp=pd.DataFrame({"rule":all_rules})
        temp["user_id"]=user_id
        temp["rank"]=[i for i in range(1,len(temp)+1)]
        temp=temp[["user_id", "rule", "rank"]]
        ss_= ss_.append(temp)
    

#ss_['rank']=[i+1 for i in ss_['rank']]
ss_.to_csv(r"C:\Users\tarun\Desktop\wyzelabs\submisssion_3.csv", index=False)





#sub_10=  ss_.groupby('user_id').head(10)
#sub_10.to_csv(r"C:\Users\tarun\Desktop\wyzelabs\sub_10.csv", index=False)


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
    if test.iloc[i].trigger_state_id not in test_trigger:
        test_trigger[test.iloc[i].trigger_state_id]= test.iloc[i].trigger_state

test_action={}
for i in range(len(test)):
    if test.iloc[i].action_id not in test_action:
        test_action[test.iloc[i].action_id]= test.iloc[i].action
    





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
hd=test_ui.sample(frac=.01)


final_ui=ui.append(test_ui).fillna(0)

from scipy.sparse.linalg import svds
import numpy as np
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


for user_id in ids:
    tt= test[test.user_id==user_id]
    print(user_id, end=" ", flush=True)
    temp=pd.DataFrame()
    #break
    new_user_latent_factors = u[ui_reconstructed.index.get_loc(user_id)]
    #new_user_latent_factors=np.array(ui_reconstructed.loc["test_4"].values).reshape(-1)
    predicted_ratings = np.dot(new_user_latent_factors, v)
    
    #top_n = 10
    #recommended_item_indices = np.argsort(predicted_ratings)[-top_n:][::-1]
    recommended_item_indices = np.argsort(predicted_ratings)[::-1]
    recommended_items = ui.columns[recommended_item_indices]
    # Get the predicted ratings for the top N items
    top_n_predicted_ratings = predicted_ratings[recommended_item_indices]

    preds={}
    for i,j in zip(recommended_items,top_n_predicted_ratings):
        if i not in preds:
            preds[i]=j
        else:
            preds[i]+=j
    preds = dict(sorted(preds.items(), key=lambda item: item[1],reverse=True))
    #preds=dict(list(preds.items())[:10])
    preds={i:j for i,j in preds.items() if i not in list(tt["item"])}


    top_quartile_cutoff = np.percentile([i for i in preds.values()], 75)

    preds={i:j for i,j in preds.items() if j >=top_quartile_cutoff}
    recommended_items=[i for i in preds]
    # Get top-N recommended items
    #num_recommendations = 1000
    #recommended_item_indices = np.argsort(predicted_ratings)[::-1]#[:num_recommendations]
    #recommended_items = final_ui.columns[recommended_item_indices]
    #check which ones are applicable to the uesr_id
    
    listed_devices= test[test.user_id==user_id]
    #listed_devices.trigger_device
    listed_devices=list(set(list(listed_devices.trigger_device)+list(listed_devices.action_device)))
    
    #recommended_items=recommended_items[[True if i.split("_")[0] in listed_devices and i.split("_")[3] in listed_devices  else False for i in recommended_items]]
    recommended_items=[i for i in recommended_items if i.split("_")[0] in listed_devices and i.split("_")[3] in listed_devices]
    
    if recommended_items!=[]:
        #recommended_item=pd.DataFrame(recommended_items[0:50]).reset_index()
        recommended_item=pd.DataFrame(recommended_items).reset_index()
        
        recommended_item.columns= ["rank", "rule"]
        recommended_item["user_id"]=user_id
        recommended_item=recommended_item[["user_id", "rule", "rank"]]
        
        
        recommended_item[['trigger_device', 'trigger', 'action', 'action_device']] = recommended_item['rule'].str.split('_', expand=True)
        
        recommended_item["trigger_device_id"]=[[i for i,j in test_device[user_id].items() if j==k] for k in recommended_item["trigger_device"]]
        
        
        recommended_item["action_device_id"]=[[i for i,j in test_device[user_id].items() if j==k] for k in recommended_item["action_device"]]
        
        
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
            multi_index = pd.MultiIndex.from_product([trigger_device, trigger, action,action_device ])
            
            rules= [str(i[0])+"_"+str(i[1])+"_"+str(i[2])+"_"+str(i[3]) for i in multi_index]
            all_rules+=rules
        all_rules= all_rules[:50]
        
        temp=pd.DataFrame({"rule":all_rules})
        temp["user_id"]=user_id
        temp["rank"]=[i for i in range(1,len(temp)+1)]
        temp=temp[["user_id", "rule", "rank"]]
        ss_= ss_.append(temp)
    

#ss_['rank']=[i+1 for i in ss_['rank']]
ss_.to_csv(r"C:\Users\tarun\Desktop\wyzelabs\submisssion_3.csv", index=False)





#sub_10=  ss_.groupby('user_id').head(10)
#sub_10.to_csv(r"C:\Users\tarun\Desktop\wyzelabs\sub_10.csv", index=False)


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
    if test.iloc[i].trigger_state_id not in test_trigger:
        test_trigger[test.iloc[i].trigger_state_id]= test.iloc[i].trigger_state

test_action={}
for i in range(len(test)):
    if test.iloc[i].action_id not in test_action:
        test_action[test.iloc[i].action_id]= test.iloc[i].action
    





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
hd=test_ui.sample(frac=.01)


final_ui=ui.append(test_ui).fillna(0)

from scipy.sparse.linalg import svds
import numpy as np
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


for user_id in ids:
    tt= test[test.user_id==user_id]
    print(user_id, end=" ", flush=True)
    temp=pd.DataFrame()
    #break
    new_user_latent_factors = u[ui_reconstructed.index.get_loc(user_id)]
    #new_user_latent_factors=np.array(ui_reconstructed.loc["test_4"].values).reshape(-1)
    predicted_ratings = np.dot(new_user_latent_factors, v)
    
    #top_n = 10
    #recommended_item_indices = np.argsort(predicted_ratings)[-top_n:][::-1]
    recommended_item_indices = np.argsort(predicted_ratings)[::-1]
    recommended_items = ui.columns[recommended_item_indices]
    # Get the predicted ratings for the top N items
    top_n_predicted_ratings = predicted_ratings[recommended_item_indices]

    preds={}
    for i,j in zip(recommended_items,top_n_predicted_ratings):
        if i not in preds:
            preds[i]=j
        else:
            preds[i]+=j
    preds = dict(sorted(preds.items(), key=lambda item: item[1],reverse=True))
    #preds=dict(list(preds.items())[:10])
    preds={i:j for i,j in preds.items() if i not in list(tt["item"])}


    top_quartile_cutoff = np.percentile([i for i in preds.values()], 75)

    preds={i:j for i,j in preds.items() if j >=top_quartile_cutoff}
    recommended_items=[i for i in preds]
    # Get top-N recommended items
    #num_recommendations = 1000
    #recommended_item_indices = np.argsort(predicted_ratings)[::-1]#[:num_recommendations]
    #recommended_items = final_ui.columns[recommended_item_indices]
    #check which ones are applicable to the uesr_id
    
    listed_devices= test[test.user_id==user_id]
    #listed_devices.trigger_device
    listed_devices=list(set(list(listed_devices.trigger_device)+list(listed_devices.action_device)))
    
    #recommended_items=recommended_items[[True if i.split("_")[0] in listed_devices and i.split("_")[3] in listed_devices  else False for i in recommended_items]]
    recommended_items=[i for i in recommended_items if i.split("_")[0] in listed_devices and i.split("_")[3] in listed_devices]
    
    if recommended_items!=[]:
        #recommended_item=pd.DataFrame(recommended_items[0:50]).reset_index()
        recommended_item=pd.DataFrame(recommended_items).reset_index()
        
        recommended_item.columns= ["rank", "rule"]
        recommended_item["user_id"]=user_id
        recommended_item=recommended_item[["user_id", "rule", "rank"]]
        
        
        recommended_item[['trigger_device', 'trigger', 'action', 'action_device']] = recommended_item['rule'].str.split('_', expand=True)
        
        recommended_item["trigger_device_id"]=[[i for i,j in test_device[user_id].items() if j==k] for k in recommended_item["trigger_device"]]
        
        
        recommended_item["action_device_id"]=[[i for i,j in test_device[user_id].items() if j==k] for k in recommended_item["action_device"]]
        
        
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
            multi_index = pd.MultiIndex.from_product([trigger_device, trigger, action,action_device ])
            
            rules= [str(i[0])+"_"+str(i[1])+"_"+str(i[2])+"_"+str(i[3]) for i in multi_index]
            all_rules+=rules
        all_rules= all_rules[:50]
        
        temp=pd.DataFrame({"rule":all_rules})
        temp["user_id"]=user_id
        temp["rank"]=[i for i in range(1,len(temp)+1)]
        temp=temp[["user_id", "rule", "rank"]]
        ss_= ss_.append(temp)
    

#ss_['rank']=[i+1 for i in ss_['rank']]
ss_.to_csv(r"C:\Users\tarun\Desktop\wyzelabs\submisssion_3.csv", index=False)





#sub_10=  ss_.groupby('user_id').head(10)
#sub_10.to_csv(r"C:\Users\tarun\Desktop\wyzelabs\sub_10.csv", index=False)


