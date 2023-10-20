# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 19:26:07 2023

@author: tarun
"""
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



train['item']= [i+"_"+k+"_"+l+"_"+m for i,k,l,m in zip(train.trigger_device,train.trigger_state, train.action,train.action_device, )]
test['item']= [i+"_"+k+"_"+l+"_"+m for i,k,l,m in zip(test.trigger_device,test.trigger_state,  test.action,test.action_device)]


train["c1"]=[i+"_"+k for i,k in zip(train.trigger_device,train.action_device, )]
train["c2"]=[i+"_"+k for i,k in zip(train.trigger_state, train.action )]
mat= train[["c1", "c2"]]
mat["strength"]=1
op=mat.groupby(["c1", "c2"])["strength"].count().reset_index()

ui= op.pivot(index="c1", columns="c2", values="strength").fillna(0)



# =============================================================================
# 
# =============================================================================


#cerate vocab
kl=train.groupby(["trigger_device_id","action_device_id",])["item"].apply(lambda x: list(x))

sentences=[", ".join(i) for i in kl]


# Split sentences into words
word_sequences = [sentence.split(",") for sentence in sentences]

# Flatten the word sequences into a list of words
words = [word for sequence in word_sequences for word in sequence]

# Create a vocabulary of unique words
vocabulary = set(words)

# Calculate word probabilities
word_probabilities = {word: words.count(word) / len(words) for word in vocabulary}

# Function to predict missing word
def predict_missing_word(sentence):
    words = sentence.split(",")
    c1= words[0].split("_")[0]+"_"+words[0].split("_")[3]
    #use ui to remove all the trig_action which can not happen
    vocabulary_= list(ui.loc[c1][ui.loc[c1]>0].index.values)
    vocabulary_=[words[0].split("_")[0]+"_"+i+"_"+words[0].split("_")[3] for i in vocabulary_]
    missing_word_probabilities = []

    for word in vocabulary_:
        #break
        new_sentence = ', '.join(words + [word])
        probability = 1  # Initialize probability for this word
        for w in new_sentence.split(", "):
            probability *= word_probabilities.get(w, 0.000001)  # Add Laplace smoothing
        missing_word_probabilities.append((word, probability))
        
        klo=sorted(missing_word_probabilities, key=lambda x: x[1])[::-1]
    
    return klo



# Example usage
#sentence = "ContactSensor_Has been open for_Unmute Notifications_Cloud"
#missing_word, probability = predict_missing_word(sentence)

#print(f"The missing word is '{missing_word}' with probability {probability:.5f}")




#lets do the testing!!


ids= set([i for i in test.user_id])
ss_= pd.DataFrame()

for num, id_ in enumerate(ids):
    print(num, end=" ", flush=True)
    #break
    tt= test[test.user_id==id_]
    
    new_tt=tt.groupby(["trigger_device_id","action_device_id",])["item"].apply(lambda x: list(x)).reset_index()
    rules=[]
    for cntr in range(len(new_tt)):
        #break
        trig,_,_,act= new_tt['item'].iloc[0][0].split("_")

        
        #break
        sentence= ", ".join(new_tt["item"].iloc[cntr])
        missing_word = predict_missing_word(sentence)
        missing_word=sorted(missing_word, key= lambda x:x[1])[0:50]
        
        missing_word=[i for i in missing_word if i[0].split("_")[0]==trig]
        missing_word=[i for i in missing_word if i[0].split("_")[3]==act]
        #[str(new_tt['trigger_device_id'].iloc[j])+"_"+str(new_tt['trigger_device_id'].iloc[j]) for i,k in missing_word]
        
        missing_word=[(i.split("_")[1:3],k) for i,k in missing_word]
        
        
        missing_word=[(str(new_tt['trigger_device_id'].iloc[cntr])+"_"+str(test_trigger[i[0]])+"_"+str(test_action[i[1]])+"_"+str(new_tt['trigger_device_id'].iloc[cntr]),k)  for i,k in missing_word]
        
        #missing_word=[(i,j)for i,j in missing_word if i.split("_")[0]== trig and i.split("_")[3]==act][0:50]
        
        rules.extend(missing_word)
        
    rules=sorted(rules, key=lambda x: x[1])[::-1][0:50]
        
    df= pd.DataFrame({"rule":[i[0] for i in rules]})
    df["user_id"]=id_
    df["rank"]=range(1, len(df)+1)
    df= df[["user_id", "rule", "rank"]][0:50]
    
    ss_=pd.concat([ss_,df ])
        
ss_.to_csv(r"C:\Users\tarun\Desktop\wyzelabs\submisssion_13.csv", index=False)
#be090cea-e0c9-4cc2-8e59-422d0de2b539
#0.0035232278936126894 lol score
    