# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 11:20:16 2023

@author: tarun
"""

import pandas as pd

train=pd.read_csv(r"C:\Users\tarun\Desktop\wyzelabs\train_rule.csv")
test= pd.read_csv(r"C:\Users\tarun\Desktop\wyzelabs\test_rule.csv")

test_action= pd.read_csv(r"C:\Users\tarun\Desktop\wyzelabs\action_map.csv")

test_action={i:j for i,j in zip(test_action.action, test_action.action_id)}


test_trigger= pd.read_csv(r"C:\Users\tarun\Desktop\wyzelabs\trigger_state_map.csv")
test_trigger={i:j for i,j in zip(test_trigger.trigger_state,test_trigger.trigger_state_id)}

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


from scipy.sparse.linalg import svds

matrix= ui.values
u, s, v = svds(matrix)
u.shape, v.shape, s.shape

matrix_reconstructed= np.dot(np.dot(u, np.diag(s)), v)
#normalize
matrix_reconstructed_norm = (matrix_reconstructed - matrix_reconstructed.min()) / (matrix_reconstructed.max() - matrix_reconstructed.min())

ui_reconstructed=pd.DataFrame(matrix_reconstructed_norm)
ui_reconstructed=ui_reconstructed.set_index(ui.index)
ui_reconstructed.columns= ui.columns



# create workable df from test data

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
test_trigger={j:i for i,j in test_trigger.items()}

test_action={}
for i in range(len(test)):
    if test.iloc[i].action_id not in test_action:
        test_action[test.iloc[i].action_id]= test.iloc[i].action
test_action={j:i for i,j in test_action.items()}





# =============================================================================
# now lets take a user and predict the closest action for each action
# then we will check if the action is already available or not
# later we will arrange them according to strength
# =============================================================================

preds={}
user_id= 19
tt= test[test.user_id==19]

for j in range(len(tt)):
    print(".", end=" ", flush=True)
    kl=tt.iloc[j]["item"]
    
    latent_feature=[1 if i == kl else 0 for i in ui.columns]
    #latent_feature=np.array(latent_feature).reshape(-1)
    
    new_user_embeddings = np.dot(latent_feature, v.T)
    predicted_ratings = np.dot(new_user_embeddings, v)
    
    top_n = 10
    recommended_item_indices = np.argsort(predicted_ratings)[-top_n:][::-1]
    recommended_items = ui.columns[recommended_item_indices]
    # Get the predicted ratings for the top N items
    top_n_predicted_ratings = predicted_ratings[recommended_item_indices]

    recommended_item_indices = np.argsort(predicted_ratings)[::-1]#[:num_recommendations]
    recommended_items = ui.columns[recommended_item_indices]
    
    recommended_items=[i for i in recommended_items if i.split("_")[0] in [kl.split("_")[0],kl.split("_")[3]] and i.split("_")[3] in [kl.split("_")[0],kl.split("_")[3]]]
    
    for i in recommended_items[:20]:
        if i not in preds:
            preds[i]=1
        else:
            preds[i]+=1



preds = dict(sorted(preds.items(), key=lambda item: item[1],reverse=True))
preds=dict(list(preds.items())[:10])
preds={i:j for i,j in preds.items() if i not in list(tt["item"])}



# =============================================================================
# 
# =============================================================================


preds={}
user_id= 19
tt= test[test.user_id==19]

for j in range(len(tt)):
    print(".", end=" ", flush=True)
    kl=tt.iloc[j]["item"]
    
    latent_feature=[1 if i == kl else 0 for i in ui.columns]
    #latent_feature=np.array(latent_feature).reshape(-1)
    
    new_user_embeddings = np.dot(latent_feature, v.T)
    predicted_ratings = np.dot(new_user_embeddings, v)

    
    #top_n = 10
    #recommended_item_indices = np.argsort(predicted_ratings)[-top_n:][::-1]
    recommended_item_indices = np.argsort(predicted_ratings)[::-1]
    recommended_items = ui.columns[recommended_item_indices]
    # Get the predicted ratings for the top N items
    top_n_predicted_ratings = predicted_ratings[recommended_item_indices]
    
    
    
    for rec,rat in zip(recommended_items,top_n_predicted_ratings):
        #break
        
        if rec.split("_")[0]==tt.iloc[j].trigger_device:
            rec=str(tt.iloc[j].trigger_device_id)+"_"+"_".join(rec.split("_")[1:])
        elif rec.split("_")[0]==tt.iloc[j].action_device:
            rec=str(tt.iloc[j].action_device_id)+"_"+"_".join(rec.split("_")[1:])
        else:
            continue
    
        if rec.split("_")[3]==tt.iloc[j].trigger_device:
            rec="_".join(rec.split("_")[:-1])+"_"+str(tt.iloc[j].trigger_device_id)
        elif rec.split("_")[3]==tt.iloc[j].action_device:
            rec="_".join(rec.split("_")[:-1])+"_"+str(tt.iloc[j].action_device_id)
        else:
            continue
        
        rec=rec.split("_")
        rec[1]=test_trigger[rec[1]]
        rec[2]=test_action[rec[2]]
        rec="_".join([str(i) for i in rec])
    
        
        if rec not in preds:
            preds[rec]=rat
        else:
            preds[rec]+=rat
    
        
preds = dict(sorted(preds.items(), key=lambda item: item[1],reverse=True)[0:50])

#preds={i:j for i,j in preds.items() if j>0}
#top_quartile_cutoff = np.percentile([i for i in preds.values()], 75)

#preds={i:j for i,j in preds.items() if j >=top_quartile_cutoff}
#preds = dict(sorted(preds.items(), key=lambda item: item[1],reverse=True)[0:50])
#preds=dict(list(preds.items())[:10])
#preds={i:j for i,j in preds.items() if i not in list(tt["item"])}



#top_quartile_cutoff = np.percentile([i for i in preds.values()], 75)

#preds={i:j for i,j in preds.items() if j >=top_quartile_cutoff}



# =============================================================================
# all
# =============================================================================




ids= set([i for i in test.user_id])

ss_= pd.DataFrame(columns= ["user_id", "rule", "rank"])

#user_id= 19
#tt= test[test.user_id==user_id]


for n,user_id in enumerate(ids):
    print(n, end=" ", flush=True)
    preds={}
    #user_id= 19
    tt= test[test.user_id==19]

    for j in range(len(tt)):
        print(".", end=" ", flush=True)
        kl=tt.iloc[j]["item"]
        
        latent_feature=[1 if i == kl else 0 for i in ui.columns]
        #latent_feature=np.array(latent_feature).reshape(-1)
        
        new_user_embeddings = np.dot(latent_feature, v.T)
        predicted_ratings = np.dot(new_user_embeddings, v)

        
        #top_n = 10
        #recommended_item_indices = np.argsort(predicted_ratings)[-top_n:][::-1]
        recommended_item_indices = np.argsort(predicted_ratings)[::-1]
        recommended_items = ui.columns[recommended_item_indices]
        # Get the predicted ratings for the top N items
        top_n_predicted_ratings = predicted_ratings[recommended_item_indices]
        
        
        
        for rec,rat in zip(recommended_items,top_n_predicted_ratings):
            #break
            
            if rec.split("_")[0]==tt.iloc[j].trigger_device:
                rec=str(tt.iloc[j].trigger_device_id)+"_"+"_".join(rec.split("_")[1:])
            elif rec.split("_")[0]==tt.iloc[j].action_device:
                rec=str(tt.iloc[j].action_device_id)+"_"+"_".join(rec.split("_")[1:])
            else:
                continue
        
            if rec.split("_")[3]==tt.iloc[j].trigger_device:
                rec="_".join(rec.split("_")[:-1])+"_"+str(tt.iloc[j].trigger_device_id)
            elif rec.split("_")[3]==tt.iloc[j].action_device:
                rec="_".join(rec.split("_")[:-1])+"_"+str(tt.iloc[j].action_device_id)
            else:
                continue
            
            rec=rec.split("_")
            rec[1]=test_trigger[rec[1]]
            rec[2]=test_action[rec[2]]
            rec="_".join([str(i) for i in rec])
        
            
            if rec not in preds:
                preds[rec]=rat
            else:
                preds[rec]+=rat
        
            
    preds = dict(sorted(preds.items(), key=lambda item: item[1],reverse=True)[0:50])
    all_rules= [i for i in preds.keys()]
    temp=pd.DataFrame({"rule":all_rules})
    temp["user_id"]=user_id
    temp["rank"]=[i for i in range(1,len(temp)+1)]
    temp=temp[["user_id", "rule", "rank"]]
    ss_= ss_.append(temp)




