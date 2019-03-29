#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 17:22:27 2019

@author: tarun.bhavnani@dev.smecorner.com
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:23:08 2019

@author: tarun.bhavnani@dev.smecorner.com
"""
import pandas as pd
import os
import re


os.chdir('/home/tarun.bhavnani@dev.smecorner.com/Desktop/ocr_trans/all mapped')
fdf= pd.read_csv("finaldf.csv")

fdf.dtypes.value_counts()
#fdf= pd.read_excel("105044_Sai Marble House_193131100000371_Final.xlsx", sheet_name="All_Transaction")
"""
fdf["counter"]=1
nm= list(fdf)

c=1
for i in os.listdir():
  print(c)
  if c<1001:
    if "xlsx" in i:
      df= pd.read_excel(i, sheet_name="All_Transaction")
      df["counter"]=c
      if list(df)==nm:
        fdf=fdf.append(df)
        c+=1
      else: 
        print("headers name not match")
    else:
      print("not xslx")
      
"""    

fdf=fdf.reset_index(drop=True)
#data['sentiment']=['pos' if (x>3) else 'neg' for x in data['stars']]
fdf["Des"] = [str(i) for i in fdf["Description"]]

fdf["Des"] = fdf["Des"].apply(lambda x: x.lower())

fdf["Des"]=[re.sub("[i|1]/w"," inwards ",i) for i in fdf["Des"]]
#fdf["Des"]=[re.sub("1/w"," inwards ",i) for i in fdf["Des"]]
fdf["Des"]=[re.sub("[o|0]/w"," outwards ",i) for i in fdf["Des"]]

#fdf["Des"]=[re.sub(r"\bdd","demandd",i) for i in fdf["Des"]]

#r'(\bdd\b|\bdd\d)' 

#fdf["Des"]=[re.sub("0/w"," outwards ",i) for i in fdf["Des"]]

fdf["Des"]=[re.sub("b/f"," brought_fwd ",i) for i in fdf["Des"]]


fdf["Des"]=[re.sub("neft"," neft ",i) for i in fdf["Des"]]

fdf["Des"]=[re.sub("[i|1]mp[s|5]"," imps ",i) for i in fdf["Des"]]

fdf["Des"]=[re.sub(r'r[i|t|1][g|8][s|5]'," rtgs ",i) for i in fdf["Des"]]
#fdf["Des"]=[re.sub("rigs"," rtgs ",i) for i in fdf["Des"]]


fdf["Des"]=[re.sub("ecs"," ecs ",i) for i in fdf["Des"]]

fdf["Des"]=[re.sub("cash"," cash ",i) for i in fdf["Des"]]

fdf["Des"]=[re.sub("nach"," nach ",i) for i in fdf["Des"]]

fdf["Des"]=[re.sub("ebank"," ebank ",i) for i in fdf["Des"]]

fdf["Des"]=[re.sub(r"c[o|0][1|l][1|l]","coll",i) for i in fdf["Des"]]# for int.co11

fdf["Des"]=[re.sub("vvdl","wdl",i) for i in fdf["Des"]]

fdf["Des"]=[re.sub("nfs"," nfs ",i) for i in fdf["Des"]]

fdf["Des"]=[re.sub(r"[1|l][o|0]an","loan",i) for i in fdf["Des"]]
#fdf["Des"]=[re.sub("10an","loan",i) for i in fdf["Des"]]


fdf["Des"]=[re.sub("[\n]|[\r]"," ",i) for i in fdf["Des"]]


fdf["Des"]=[re.sub(r"\(|\)|\[|\]"," ",i) for i in fdf["Des"]]#brackets

fdf["Des"]=[re.sub("-|:|\.|/"," ",i) for i in fdf["Des"]]

#add a c to make ac for account, here logic can be improved.
fdf["Des"]=[re.sub("a c ","ac ", i) for i in fdf["Des"]]


#for classification we stick'em together!

fdf["Des_cl"]= [re.sub(" ","",i) for i in fdf["Des"]]

#########################################################################
"Classification"

#we will Des for some cases and Des_cl for some casses for "i m p s " type cases

fdf["classification"]="Not_Tagged"


fdf["classification"]=["dd" if len(re.findall(r"\bdd\b",x))>0 else y for x,y in zip(fdf["Des"], fdf["classification"])]
#fdf["classification"].value_counts()
#fdfdd= fdf[fdf["classification"]=="transfer"]

fdf["classification"]=["ib" if len(re.findall(r"\bib\b",x))>0 else y for x,y in zip(fdf["Des"], fdf["classification"])]

fdf["classification"]=["ft" if len(re.findall(r"\bft\b",x))>0 else y for x,y in zip(fdf["Des"], fdf["classification"])]


fdf["classification"]=["brought_fwd" if len(re.findall("brought_fwd",x))>0 else y for x,y in zip(fdf["Des_cl"], fdf["classification"])]



#fdf["classification"]=["transfer" if len(re.findall("dr",x))>0 else y for x,y in zip(fdf["Des"], fdf["classification"])]
fdf["classification"]=["transfer" if len(re.findall(r"\bdr\b",x))>0 else y for x,y in zip(fdf["Des"], fdf["classification"])]


fdf["classification"]=["transfer" if len(re.findall("tpt",x))>0 else y for x,y in zip(fdf["Des_cl"], fdf["classification"])]


fdf["classification"]=["transfer" if len(re.findall("transfer",x))>0 else y for x,y in zip(fdf["Des_cl"], fdf["classification"])]




fdf["classification"]=["neft" if len(re.findall("neft",x))>0 else y for x,y in zip(fdf["Des_cl"], fdf["classification"])]

fdf["classification"]=["rtgs" if len(re.findall("rtgs",x))>0 else y for x,y in zip(fdf["Des_cl"], fdf["classification"])]


fdf["classification"]=["imps" if len(re.findall(r"[i|1]mps",x))>0 else y for x,y in zip(fdf["Des_cl"], fdf["classification"])]

fdf["classification"]=["cash" if len(re.findall("cash",x))>0 else y for x,y in zip(fdf["Des"], fdf["classification"])]

fdf["classification"]=["cash" if len(re.findall("at[m|w]",x))>0 else y for x,y in zip(fdf["Des_cl"], fdf["classification"])]

fdf["classification"]=["cash" if len(re.findall("self",x))>0 else y for x,y in zip(fdf["Des_cl"], fdf["classification"])]
#fdf["classification"]=["cash" if len(re.findall("atw",x))>0 else y for x,y in zip(fdf["Des_cl"], fdf["classification"])]
fdf["classification"]=["cash" if len(re.findall(r"\bpos\b",x))>0 else y for x,y in zip(fdf["Des"], fdf["classification"])]
#fdf["classification"]=["cash" if len(re.findall("wdl",x))>0 else y for x,y in zip(fdf["Des"], fdf["classification"])]
fdf["classification"]=["cash" if len(re.findall("debitcard",x))>0 else y for x,y in zip(fdf["Des_cl"], fdf["classification"])]


fdf["classification"]=["cheque" if len(re.findall(r"che?q",x))>0 else y for x,y in zip(fdf["Des"], fdf["classification"])]

fdf["classification"]=["cheque" if len(re.findall("cl[g|q]",x))>0 else y for x,y in zip(fdf["Des"], fdf["classification"])]

fdf["classification"]=["cheque" if len(re.findall(r"c[l|1]earing",x))>0 else y for x,y in zip(fdf["Des_cl"], fdf["classification"])]

fdf["classification"]=["cheque" if len(re.findall(r"\binf\b",x))>0 else y for x,y in zip(fdf["Des"], fdf["classification"])]


fdf["classification"]=["ecs" if len(re.findall("ecs",x))>0 else y for x,y in zip(fdf["Des_cl"], fdf["classification"])]

fdf["classification"]=["ecs" if len(re.findall("loan",x))>0 else y for x,y in zip(fdf["Des_cl"], fdf["classification"])]


fdf["classification"]=["emi" if len(re.findall(r"\bemi\b",x))>0 else y for x,y in zip(fdf["Des_cl"], fdf["classification"])]


fdf["classification"]=["nach" if len(re.findall("nach",x))>0 else y for x,y in zip(fdf["Des_cl"], fdf["classification"])]

fdf["classification"]=["nach" if len(re.findall(r"\bach\b",x))>0 else y for x,y in zip(fdf["Des"], fdf["classification"])]

fdf["classification"]=["i/w" if len(re.findall("inward",x))>0 else y for x,y in zip(fdf["Des_cl"], fdf["classification"])]
fdf["classification"]=["o/w" if len(re.findall("outward",x))>0 else y for x,y in zip(fdf["Des_cl"], fdf["classification"])]
fdf["classification"]=["int_coll" if len(re.findall("int coll",x))>0 else y for x,y in zip(fdf["Des"], fdf["classification"])]


fdf["cl_cl"]="Not_Tagged"
fdf["cl_cl"]=["charges" if len(re.findall(r"charge?",x))>0 else y for x,y in zip(fdf["Des_cl"], fdf["cl_cl"])]
fdf["cl_cl"]=["charges" if len(re.findall("chrg",x))>0 else y for x,y in zip(fdf["Des_cl"], fdf["cl_cl"])]
fdf["cl_cl"]=["charges" if len(re.findall("chgs?",x))>0 else y for x,y in zip(fdf["Des_cl"], fdf["cl_cl"])]
fdf["cl_cl"]=["charges" if len(re.findall("commission",x))>0 else y for x,y in zip(fdf["Des_cl"], fdf["cl_cl"])]
fdf["cl_cl"]=["charges" if len(re.findall(r"\bfee\b",x))>0 else y for x,y in zip(fdf["Des_cl"], fdf["cl_cl"])]


#still untagged transactions
nt= fdf[(fdf["classification"]!="Not_Tagged") | (fdf["cl_cl"]!="Not_Tagged")]
fdf.shape[0]- nt.shape[0]#nottagged




#fdf["classification"]=["return" if len(re.findall(r"retu?r?n?",x))>0 else y for x,y in zip(fdf["Des_cl"], fdf["classification"])]
#fdf["classification"]=[y+" return" if len(re.findall(r"\bret[u| ]r?n?",x))>0 else y for x,y in zip(fdf["Des"], fdf["classification"])]

#fdf["classification"]=[y+" return" if len(re.findall(r"\bretu?r?n?|return",x))>0 else y for x,y in zip(fdf["Des"], fdf["classification"])]

fdf["cl_cl"]=["return" if len(re.findall(r"\bretu?r?n?|return",x))>0 else y for x,y in zip(fdf["Des_cl"], fdf["cl_cl"])]



fdf["classification"].value_counts()
fdf["tt"].value_counts()
fdf.to_csv("finaldf8.csv")


###############################################################################
###############################################################################
###############################################################################
###############################################################################

set([i for i in fdf.classification if "return" in i])
[[x,y,z] for x,y,z in zip(newdf["Description"], newdf["classification"], newdf["tt"]) if "return" in y]


#########################################################################

'back to fdf["Des"]'

#any one of the two below will do the job of effectivly removing the digits
#1) this removes the extra spaces as well
fdf["Des"]=[" ".join([j for j in i.split() if not any(c.isdigit() for c in j)]) for i in fdf["Des"]]
#if any alpha numerics are still left

fdf["Des"]= [re.sub("[\W_]+"," ",i) for i in fdf["Des"]]



#re.sub("[\W_]+"," ",i)





"check with given transaction type"

newdf= fdf[(fdf["classification"]=="Not_Tagged")]
newdf=newdf[newdf["tt"]!="Others"]
newdf.shape

newdf1= fdf[(fdf["tt"]=="Others")]
newdf1=newdf1[newdf1["classification"]!="Not_Tagged"]
newdf1.shape




newd= fdf[(fdf["classification"]=="ecs return")]
newdf=newdf[newdf["tt"]!="Others"]
newdf.shape


newdf= fdf[(fdf["classification"]=="cheque") | (newdf["tt"]=="cheque")]

newdf.shape



########################3
"Round entries"
##########################

#list(fdf)

#for i in fdf["Credit"]:
#  print(i)
"""
fdf["round"]=0
for i in range(len(fdf)):
#  print(i)  
  #print(i,fdf.Credit[i])
  
  for j in range(i-3,i+3):
#    print(j)
    try:
        #print(fdf.Debit[j])
        if fdf.Debit[j]==fdf.Credit[i] and fdf.Credit[i]!=0 :
          print("check........ {}".format(i))
          fdf['round'][i]=1
    except:
        print("NA")

fdf['round'].value_counts()
"""
  

fdf["round"]=0
for i in range(len(fdf)):
  
  for j in range(i,i+4):
#    print(j)
    try:
        #print(fdf.Debit[j])
        if fdf.Debit[j]==fdf.Credit[i] and fdf.Credit[i]!=0 :
          print("check........ {}".format(i))
          fdf['round'][i]=1
          fdf['round'][j]=2
          if fdf["Date_new"][i]==fdf["Date_new"][j]: #for bounce
            fdf['round'][j]=3
            
    except:
        print("NA")

fdf['round'].value_counts()

#fyrther on bounce
'if round==3 and classification has "return"'

[y for x,y,z in zip(fdf["round"], fdf["Description"],fdf["classification"]) if x==3 and  "return" in z]

jk=fdf[fdf["round"]==3]["return" in fdf["classification"]]
#[i for i in fdf]


rnd= fdf[(fdf["round"]==1)|(fdf["round"]==2)|(fdf["round"]==3)]


########
"Major creditors and debitors"
########
#df7= fdf[fdf.counter==7]

from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(nb_words = 2000, split=' ',oov_token=True)
fdf["Des"] = [str(i) for i in fdf["Des"]]
tokenizer.fit_on_texts(fdf['Des'].values)
tokenizer.word_counts

high_count_words = [w for w,c in tokenizer.word_counts.items() if c >300]
#choose remove the stopwords from these
high_count_words


 #['imps','transfer','chgs','int','coll','to','loan','service','charges','ib','rtgs','charge','gst','pvt','s','m','inward','uti','clg','of','for','chq','paid','cts','no','dep','g','ft','dr','k','r','cbdt','tax','atw','tran','t','j','xxxx','p','d','nfs','h','in','cr','so','nan','inwards','return','cash','ref','c',
 # 'emi','n','nwd','x','pos','debit','card','upi','fee','paytm','sgst','cgst','by','dd','cash','cheque','fees'

#fdf.Description
stopwords=["by", "to", "ecsidr","clg","inst","trfr","from","ref","ra","of","neft","rtgs","transfer","tpt","at","cash","cam",
           "nfs", "nach","ach","gst","rs","fee","co","clg","service","xxxxxx","cr","ecs","ubi","bob",'imps','chgs','int','coll','to','loan','service','charges','ib','rtgs','charge','gst','pvt','s','m','inward','uti','clg','of','for','chq','paid','cts','no','dep','g','ft','dr','k','r','cbdt','tax','atw','tran','t','j','xxxx','p','d','nfs','h','in','cr','so','nan','inwards','return','cash','ref','c',
  'emi','n','nwd','x','pos','debit','card','upi','fee','paytm','sgst','cgst','by','dd','cash','cheque','fees']


#df7['des']= [re.sub(r'[^a-zA-Z]'," ",i) for i in df7['Des']] 
fdf["Des1"]= [" ".join([j for j in i.split() if j not in stopwords]) for i in fdf["Des"]] 

from fuzzywuzzy import fuzz
from fuzzywuzzy import process


narrations = list(fdf["Des1"][1:1000])
list_groups = []
group_count = 0

for i,string in enumerate(narrations):
  #print(i,string)
    match_list = process.extract(string, narrations, scorer = fuzz.token_set_ratio, limit = len(narrations))     
    match_list = [ele[0] for ele in match_list if ele[1] >75]
    for ele in match_list:
      narrations.remove(ele)
      print("a")
    if len(match_list) > 5: #in list(range(2, 10)):
        list_groups.append(tuple((match_list, len(match_list), group_count)))
        print("b")
        group_count +=1


df = pd.DataFrame(list_groups, columns=['Strings', 'count', 'group'])    
df.to_excel("testing1.xlsx", index = False)



"Creditors"

fdf_cr= fdf[fdf.Debit==0]

#fdf["stem_des"]=[stemmer.stem(t) for t in fdf["Des"]]

fdf_cr.des.value_counts()  [0:10]

#stemmed
fdf_cr["stem"]=[" ".join([j[0:4] for j in i.split()]) for i in fdf_cr["des"]]
fdf_cr.stem.value_counts()  [0:10]


"Debitors"

fdf_db= fdf[fdf.Credit==0]

fdf_db.des.value_counts()[0:10]  

#here we can do a simple bag of words clustering on the "Des" to cluster
#why: "neft ethnic weaves emporiumpriv" and  "rtgs ethnic weavesemporiu" should come together

"""
from keras.preprocessing.text import Tokenizer


tok = Tokenizer(num_words=100)
tok.fit_on_texts(fdf["Des"].values)
X=tok.texts_to_sequences(fdf["Des"].values)
"""


from sklearn.feature_extraction.text import TfidfVectorizer




#monthly patterns

fdf["n"]=1
list(fdf)
fdf2=fdf.groupby(['des'])[["Credit", "Debit","n"]].sum()

#can se "int"
#193131100000371:Int.Co11:01-12-2017 to 31-12-2017

"see what to do with special characters if its a : or a ."



ghp=fdf[fdf['Des'].str.contains("int")]







##########Month;y Patterns
list(fdf)
#df['date'] = pd.to_datetime(df['date'])
#df.groupby(df['date'].dt.strftime('%B'))['Revenue'].sum().sort_values()

fdf['Date_new']=pd.to_datetime(fdf['Date_new'])

#monthwise

op=fdf.groupby(fdf['Date_new'].dt.strftime('%B'))['Credit'].sum()
jk=fdf.groupby(fdf['Date_new'].dt.strftime('%B'))['Debit'].sum()

kl=pd.concat([op,jk], axis=1)
#pd.DataFrame({'r': r, 's': s})

#daywise

d1=fdf.groupby(fdf['Date_new'].dt.strftime('%d'))['Credit'].sum()
d2=fdf.groupby(fdf['Date_new'].dt.strftime('%d'))['Debit'].sum()
d12=pd.concat([d1,d2], axis=1)




##############Bounce find out

list(fdf)
fdf['Balance']

df['col2'] = df.apply(lambda x: x['col1'] if x['col1'] in l else None, axis=1)

fdf["Bounce_check"]=fdf.apply(lambda x: True if x['Balance']>0 else False, axis=1)
fdf["Bounce_check"].value_counts()




