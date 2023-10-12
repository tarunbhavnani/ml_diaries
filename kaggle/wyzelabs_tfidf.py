# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 11:59:54 2023

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

# test_devices={}
# for i in range(len(test)):
    
#     test_devices[test.trigger_device_id.iloc[i]]=test.trigger_device.iloc[i]
#     test_devices[test.action_device_id.iloc[i]]=test.action_device.iloc[i]

test_device={}
for j in list(set(test.user_id)):
    tt= test[test.user_id==j]
    td={}
    for i in range(len(tt)):
        
            td[tt.iloc[i].trigger_device]= str(tt.iloc[i].trigger_device_id)
        
            td[tt.iloc[i].action_device]= str(tt.iloc[i].action_device_id)
    
    test_device[j]=td

test_trigger={}
for i in range(len(test)):
    if test.iloc[i].trigger_state_id not in test_trigger:
        test_trigger[str(test.iloc[i].trigger_state_id)]= str(test.iloc[i].trigger_state)
test_trigger={j:i for i,j in test_trigger.items()}

test_action={}
for i in range(len(test)):
    if test.iloc[i].action_id not in test_action:
        test_action[str(test.iloc[i].action_id)]= str(test.iloc[i].action)
test_action={j:i for i,j in test_action.items()}

# =============================================================================


train['item']= [i+"_"+k+"_"+l+"_"+m for i,k,l,m in zip(train.trigger_device,train.trigger_state, train.action,train.action_device )]
test['item']= [i+"_"+k+"_"+l+"_"+m for i,k,l,m in zip(test.trigger_device,test.trigger_state, test.action,test.action_device )]

df= train[["user_id", "item"]].append(test[["user_id", "item"]])




# =============================================================================
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Define a custom tokenizer function
def custom_tokenizer(text):
    # Split the text on commas and return individual words as tokens
    return text.split('_')

# Create a TfidfVectorizer with the custom tokenizer
vectorizer = TfidfVectorizer(
    analyzer='word',
#    ngram_range=(1, 2),
#    min_df=0.003,
#    max_df=0.5,
    #max_features=5000,
    stop_words="english",
    tokenizer=custom_tokenizer
)

vectorizer.fit(df.item.values)


X= vectorizer.transform(df.item.values)


# =============================================================================
# 
# =============================================================================
user_id=19

related_content=list(set(df[df["user_id"]==19]["item"]))


#related_text=df[df["contentId"].isin(related_content)]["text"].reset_index()

#lets find similar stuff
#rated_text=pd.DataFrame([0]*len(articles_df),columns=["score"])
rated_text=df.copy()
rated_text["score"]=0
for text in related_content:
    #break
    #text=related_content.text.iloc[num]
    text_vec= vectorizer.transform([text])
    scores= cosine_similarity(X,text_vec)
    scores=[i[0] for i in scores]
    rated_text["score"]+=scores

#remove retalted text
rated_text=rated_text[~rated_text['item'].isin(related_content)]
rated_text=rated_text.groupby("item")["score"].apply(lambda x : np.mean(x)).reset_index()
rated_text=rated_text[rated_text.score>0]




# =============================================================================
# one by one
# =============================================================================

df= train[["user_id", "item","trigger_device","trigger_state","action","action_device"]].append(test[["user_id", "item","trigger_device","trigger_state","action","action_device"]])
rated_text=df.copy()

user_id=19

tt= test[test.user_id==19]
tt=tt.groupby(['user_id',
 'trigger_device',
 'trigger_device_id',
 'trigger_state',
 'trigger_state_id',
 'action',
 'action_id',
 'action_device',
 'action_device_id', "item"]).count().reset_index()

#user_items=list(set(df[df["user_id"]==19]["item"]))



# #i=0
# all_rules=pd.DataFrame()
# for i in range(len(tt)):
#     rated_text=df.copy()
#     #inst= tt.iloc[i]
#     user_item=tt.iloc[i]['item']
#     text_vec= vectorizer.transform([user_item])
#     scores= cosine_similarity(X,text_vec)
#     scores=[i[0] for i in scores]
#     rated_text["score"]=scores
#     listed_devices= [user_item.split("_")[0],user_item.split("_")[3]]
    
    
    
#     rated_text=rated_text[[True if i in listed_devices and j in listed_devices else False  for i,j in zip(rated_text.action_device, rated_text.trigger_device)]]
    
    
#     rated_text=rated_text[rated_text.item!=user_item]
#     rated_text=rated_text[rated_text.score>0]
#     #rated_text["rule"]= [str(i)+"_"+str(k)+"_"+str(l)+"_"+str(m) for i,k,l,m in zip(rated_text.trigger_device_id,rated_text.trigger_state_id, rated_text.action_id,rated_text.action_device_id )]
    
#     #rated_text=rated_text.sort_values(by="score", ascending=False)
#     #kl=rated_text.iloc[0:100]
    
#     rated_text=rated_text.groupby(["trigger_device","trigger_state","action","action_device"])["score"].apply(lambda x : np.mean(x)).reset_index()
#     rated_text["trigger_device"]=rated_text.trigger_device.map(test_device[user_id])
#     rated_text["trigger_state"]=rated_text.trigger_state.map(test_trigger)
#     rated_text["action"]=rated_text.action.map(test_action)
#     rated_text["action_device"]=rated_text.action_device.map(test_device[user_id])
#     rated_text=rated_text.dropna()
#     rated_text["rule"]=[str(int(i))+"_"+str(int(j))+"_"+str(int(k))+"_"+str(int(l)) for i,j,k,l in zip(rated_text["trigger_device"],rated_text["trigger_state"],rated_text["action"],rated_text["action_device"])]
    
#     rated_text=rated_text.sort_values(by="score", ascending=False)[0:50]
#     #rated_text["user_id"]=user_id
#     all_rules=all_rules.append(rated_text[["rule", "score"]])
    
# all_rules=all_rules.groupby("rule")["score"].sum().reset_index()    
    
    
ids= set([i for i in test.user_id])

ss_= pd.DataFrame(columns= ["user_id", "rule", "rank"])
    

for n,user_id in enumerate(ids):
    print(n, end=" ", flush=True)
    tt= test[test.user_id==user_id]
    all_rules_data = []
    
    for i in range(len(tt)):
        user_item = tt.iloc[i]['item']
        listed_devices = [user_item.split("_")[0], user_item.split("_")[3]]
    
        
        # Calculate cosine similarity
        text_vec = vectorizer.transform([user_item])
        scores = cosine_similarity(X, text_vec).flatten()
    
        # Add scores to the filtered DataFrame
        df['score'] = scores
        
        # Filter the DataFrame based on the listed devices
        filtered_df = df[df['action_device'].isin(listed_devices) & df['trigger_device'].isin(listed_devices)]
    
        # Exclude rows with the same item and where score is less than or equal to 0
        filtered_df = filtered_df[(filtered_df['item'] != user_item) & (filtered_df['score'] > 0)]
    
        # Group and aggregate by rule and calculate the mean score
        aggregated_df = filtered_df.groupby(["trigger_device", "trigger_state", "action", "action_device"])['score'].mean().reset_index()
    
        # Map device and action values
        aggregated_df["trigger_device"] = aggregated_df["trigger_device"].map(test_device[user_id])
        aggregated_df["trigger_state"] = aggregated_df["trigger_state"].map(test_trigger)
        aggregated_df["action"] = aggregated_df["action"].map(test_action)
        aggregated_df["action_device"] = aggregated_df["action_device"].map(test_device[user_id])
    
        # Drop rows with NaN values and format the 'rule' column
        aggregated_df = aggregated_df.dropna()
        aggregated_df["rule"] = aggregated_df[["trigger_device","trigger_state","action","action_device"]].apply(lambda row: "_".join(map(str, row)), axis=1)
    
        # Sort by score in descending order and select the top 50 rows
        aggregated_df = aggregated_df.sort_values(by="score", ascending=False).head(50)
    
        # Append data to the list
        all_rules_data.append(aggregated_df[['rule', 'score']])
    
    # Concatenate the list of DataFrames into a single DataFrame
    all_rules = pd.concat(all_rules_data, ignore_index=True)
    
    # Group by 'rule' and sum the scores
    all_rules = all_rules.groupby("rule")["score"].sum().reset_index().sort_values(by="score", ascending=False)
    all_rules["user_id"]=user_id
    all_rules["rank"]=range(1,len(all_rules)+1)
    all_rules=all_rules[["user_id", "rule", "rank"]][:50]
    
    ss_=pd.concat([ss_, all_rules])














