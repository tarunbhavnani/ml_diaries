# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 17:15:53 2023

@author: tarun
"""

import pandas as pd

train=pd.read_csv(r"C:\Users\tarun\Desktop\wyzelabs\train_rule.csv")
test= pd.read_csv(r"C:\Users\tarun\Desktop\wyzelabs\test_rule.csv")

test_action= pd.read_csv(r"C:\Users\tarun\Desktop\wyzelabs\action_map.csv")

test_action={i:j for i,j in zip(test_action.action, test_action.action_id)}


test_trigger= pd.read_csv(r"C:\Users\tarun\Desktop\wyzelabs\trigger_state_map.csv")
test_trigger={i:j for i,j in zip(test_trigger.trigger_state,test_trigger.trigger_state_id)}


#test_devices=pd.read_csv(r"C:\Users\tarun\Desktop\wyzelabs\test_device.csv")

#test_devices={i:j for i,j in zip(test_devices.device_id, test_devices.device_model)}

test_devices={}
for i in range(len(test)):
    
    test_devices[test.trigger_device_id.iloc[i]]=test.trigger_device.iloc[i]
    test_devices[test.action_device_id.iloc[i]]=test.action_device.iloc[i]
    


# =============================================================================
# 
# =============================================================================
hd= train.head()

train['item']= [i+"_"+k+"_"+l+"_"+m for i,k,l,m in zip(train.trigger_device,train.trigger_state, train.action,train.action_device )]

#Create a ui matrix
train["count"]=1



grouped_df_train = train.groupby(['user_id', 'item']).size().reset_index(name='count')


test['item']= [i+"_"+k+"_"+l+"_"+m for i,k,l,m in zip(test.trigger_device,test.trigger_state, test.action,test.action_device )]

#Create a ui matrix
test["count"]=1



# =============================================================================

grouped_df_test = test.groupby(['user_id', 'item']).size().reset_index(name='count')

grouped_df= grouped_df_train.append(grouped_df_test)




#user-item
ui= grouped_df.pivot(index="user_id", columns="item", values="count").fillna(0)


#item-item
from sklearn.metrics.pairwise import cosine_similarity

ii=pd.DataFrame(cosine_similarity(ui.T,ui.T))

from sklearn.preprocessing import MinMaxScaler

# Create a MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the similarity matrix
ii_normalized = scaler.fit_transform(ii)

# Convert the result back to a DataFrame
ii_normalized = pd.DataFrame(ii_normalized, columns=ii.columns, index=ii.index)

#ii.index= list(ui)
#ii.columns= list(ui)
#lets ee user -1479311724257856983

user_id=19

user= ui.loc[user_id]


cont_=user[user>1]


for n,con in enumerate(cont_.index):
    #break
    if n==0:
        cont=ii.loc[con]
    else:
        cont+=ii.loc[con]
    


#cont=ii.loc[cont_]
cont=pd.DataFrame(cont).reset_index()
cont.columns= ["item", "score"]
cont= cont.sort_values(by="score", ascending=False)[0:1000]


cont['devices']=[(i.split("_")[0],i.split("_")[3]) for i in cont['item']]
cont['devices']=[(i.split("_")[0], i.split("_")[3]) if i.split("_")[0] <= i.split("_")[3] else (i.split("_")[3], i.split("_")[0]) for i in cont['item']]

#assign values to device action and trigger names

tt= test[test.user_id==user_id]
tt['devices']=[(i.split("_")[0], i.split("_")[3]) if i.split("_")[0] <= i.split("_")[3] else (i.split("_")[3], i.split("_")[0]) for i in tt['item']]
available_groups= list(set([(i,j) for i,j in tt['devices']]))

cont=cont[cont['devices'].isin(available_groups)]



# =============================================================================
# use users as grouped by trigger and action device
# =============================================================================


grouped_df_test=test.groupby(["user_id","trigger_device_id", "action_device_id", "item" ]).size().reset_index(name='count')
grouped_df_test['user_id']=[str(i)+"_"+str(j)+"_"+str(k) for i,j,k in zip(grouped_df_test.user_id, grouped_df_test.trigger_device_id, grouped_df_test.action_device_id)]
grouped_df_test.drop(["trigger_device_id","action_device_id"], axis=1, inplace=True)


grouped_df= grouped_df_train.append(grouped_df_test)




#user-item
ui= grouped_df.pivot(index="user_id", columns="item", values="count").fillna(0)


#item-item
from sklearn.metrics.pairwise import cosine_similarity

ii=pd.DataFrame(cosine_similarity(ui.T,ui.T))

ii.index= list(ui)
ii.columns= list(ui)
#lets ee user -1479311724257856983

user_id=19

user_= ui.loc[[i for i in grouped_df_test.user_id if i.split("_")[0]==str(user_id)]]


user_preds=pd.DataFrame()
for i in range(len(user_)):
    try:
        #break
        user= user_.iloc[i]
        uid=user_.index[i]
        
        devs= [test_devices[int(uid.split("_")[1])],test_devices[int(uid.split("_")[2])]]
        
        devs={test_devices[i]:i for i in [int(uid.split("_")[1]),int(uid.split("_")[2])]}
        
        
        cont_=user[user>0]
        
        for n,con in enumerate(cont_.index):
            #break
            if n==0:
                cont=ii.loc[con]
            else:
                cont+=ii.loc[con]
    except Exception as e:
        print(e)
    
        
    
    
    #[i for i,j in zip(cont, cont.index)]
    
    cont=pd.DataFrame(cont).reset_index()
    cont.columns= ["item", "score"]
    cont=cont[(cont.score>0) & (cont.score<.99)]
    cont=cont[[True if i.split("_")[0] in devs and i.split("_")[3] in devs else False for i in cont.item]]
    
    cont["item"]=[i.split("_") for i in cont.item]
    cont["item"]=["_".join([str(devs[i[0]]), str(test_trigger[i[1]]),str(test_action[i[2]]), str(devs[i[3]])])  for i in cont["item"]]
    
    #user_preds=user_preds.append(cont)
    user_preds=pd.concat([user_preds, cont])
    
user_preds=user_preds.groupby("item")["score"].sum().reset_index()
# Sort 
user_preds = user_preds.sort_values(by='score', ascending=False)[0:50]

# Create a new column 'rank' with sequential values starting from 1
user_preds['rank'] = range(1, len(user_preds) + 1)
user_preds["user_id"]=user_id
user_preds=user_preds[["user_id", "item", "rank"]]
user_preds.columns=["user_id", "rule", "rank"]



    
# =============================================================================
# now do for all users
# =============================================================================

ids= set([i for i in test.user_id])

ss_= pd.DataFrame(columns= ["user_id", "rule", "rank"])
    

not_processed=[]
for user_id in ids:
    try:
        print(user_id, end=" ", flush=True)
        user_= ui.loc[[i for i in grouped_df_test.user_id if i.split("_")[0]==str(user_id)]]
        
        
        user_preds=pd.DataFrame()
        for i in range(len(user_)):
            try:
                #break
                user= user_.iloc[i]
                uid=user_.index[i]
                
                #devs= [test_devices[int(uid.split("_")[1])],test_devices[int(uid.split("_")[2])]]
                
                devs={test_devices[i]:i for i in [int(uid.split("_")[1]),int(uid.split("_")[2])]}
                
                
                cont_=user[user>0]
                
                for n,con in enumerate(cont_.index):
                    #break
                    if n==0:
                        cont1=ii_normalized.loc[con]
                    else:
                        cont1+=ii_normalized.loc[con]
                
                
                
                #[i for i,j in zip(cont, cont.index)]
                
                cont=pd.DataFrame(cont1).reset_index()
                cont.columns= ["item", "score"]
                #cont=cont[(cont.score>0) & (cont.score<.99)]
                cont=cont[[True if i.split("_")[0] in devs and i.split("_")[3] in devs else False for i in cont.item]]
                
                cont["item"]=[i.split("_") for i in cont.item]
                cont["item"]=["_".join([str(devs[i[0]]), str(test_trigger[i[1]]),str(test_action[i[2]]), str(devs[i[3]])])  for i in cont["item"]]
                
                #user_preds=user_preds.append(cont)
                user_preds=pd.concat([user_preds, cont])
            except Exception as e:
                print(e)
            
        user_preds=user_preds.groupby("item")["score"].sum().reset_index()
        # Sort 
        user_preds = user_preds.sort_values(by='score', ascending=False)[0:50]
        
        # Create a new column 'rank' with sequential values starting from 1
        user_preds['rank'] = range(1, len(user_preds) + 1)
        user_preds["user_id"]=user_id
        user_preds=user_preds[["user_id", "item", "rank"]]
        user_preds.columns=["user_id", "rule", "rank"]
        
# =============================================================================
#         #remove the rules already present
# =============================================================================
        
        
        ss_=pd.concat([ss_, user_preds])
    except:
        not_processed.append(user_id)
        
assert(len(set(ss_.user_id))==len(set(test.user_id)))

ss_.to_csv(r"C:\Users\tarun\Desktop\wyzelabs\submisssion_6.csv", index=False)

#score .006, v bad
