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











# =============================================================================
# all
# =============================================================================





ids= set([i for i in test.user_id])

ss_= pd.DataFrame(columns= ["user_id", "rule", "rank"])
dfs_to_concat = []
#user_id= 19
#tt= test[test.user_id==user_id]




#latent_features= [[1 if i == tt.iloc[j]["item"] else 0 for i in ui.columns] for j in range(len(tt))]

#ne=np.dot(latent_features, v.T)
#ne.shape
for n,user_id in enumerate(ids):
    print(n, end=" ", flush=True)
    preds={}
    #user_id= 19
    tt= test[test.user_id==user_id]
    latent_features= [[1 if i == tt.iloc[j]["item"] else 0 for i in ui.columns] for j in range(len(tt))]
    ne=np.dot(latent_features, v.T)
    pr = np.dot(ne, v)

    for j in range(len(tt)):
        print(".", end=" ", flush=True)
        #kl=tt.iloc[j]["item"]
        
        #latent_feature=[1 if i == kl else 0 for i in ui.columns]
        #latent_feature=np.array(latent_feature).reshape(-1)
        
        #new_user_embeddings = np.dot(latent_feature, v.T)
        #predicted_ratings = np.dot(ne[j], v)

        
        #top_n = 10
        #recommended_item_indices = np.argsort(predicted_ratings)[-top_n:][::-1]
        recommended_item_indices = np.argsort(pr[j])[::-1]
        recommended_items = ui.columns[recommended_item_indices][:100]
        # Get the predicted ratings for the top N items
        top_n_predicted_ratings = pr[j][recommended_item_indices][:100]
        
        
        
        
        for rec,rat in zip(recommended_items,top_n_predicted_ratings):
            
            p1,p2,p3,p4= rec.split("_")
            #break
            
            if p1==tt.iloc[j].trigger_device:
                p1=str(tt.iloc[j].trigger_device_id)
            elif p1==tt.iloc[j].action_device:
                p1=str(tt.iloc[j].action_device_id)
            else:
                continue
        
            if p4==tt.iloc[j].trigger_device:
                p4=str(tt.iloc[j].trigger_device_id)
            elif p4==tt.iloc[j].action_device:
                p4=str(tt.iloc[j].action_device_id)
            else:
                continue
            
            #rec=rec.split("_")
            p2=test_trigger[p2]
            p3=test_action[p3]
            rec="_".join([str(i) for i in [p1,p2,p3,p4]])
        
            
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
    dfs_to_concat.append(temp)
    


ss_ = pd.concat(dfs_to_concat, ignore_index=True)

assert(len(set(ss_.user_id))==len(set(test.user_id)))

ss_.to_csv(r"C:\Users\tarun\Desktop\wyzelabs\submisssion_5.csv", index=False)


