
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:23:08 2019

@author: tarun.bhavnani@dev.smecorner.com
"""
import pandas as pd
import os
import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from time import time

def clean_transc(dat):
  #have removed the spaces from imps and rtgs as in neft, see
  t0= time()
  dat["Des"] = [str(i) for i in dat["Description"]]
  dat["Des"] = dat["Des"].apply(lambda x: x.lower())
  dat["Des"]=[re.sub("[i|1]/w"," inwards ",i) for i in dat["Des"]]
  dat["Des"]=[re.sub("[o|0]/w"," outwards ",i) for i in dat["Des"]]
  dat["Des"]=[re.sub("b/f"," brought_fwd ",i) for i in dat["Des"]]
  #dat["Des"]=[re.sub("neft"," neft ",i) for i in dat["Des"]]
  dat["Des"]=[re.sub("[i|1]mp[s|5]","imps",i) for i in dat["Des"]]
  dat["Des"]=[re.sub(r'r[i|t|1][g|8][s|5]',"rtgs",i) for i in dat["Des"]]
  dat["Des"]=[re.sub("ecs"," ecs ",i) for i in dat["Des"]]
  dat["Des"]=[re.sub("cash"," cash ",i) for i in dat["Des"]]
  dat["Des"]=[re.sub("nach","nach",i) for i in dat["Des"]]
  dat["Des"]=[re.sub("ebank","ebank",i) for i in dat["Des"]]
  dat["Des"]=[re.sub(r"c[o|0][1|l][1|l]","coll",i) for i in dat["Des"]]# for int.co11
  dat["Des"]=[re.sub("vvdl","wdl",i) for i in dat["Des"]]
  dat["Des"]=[re.sub("nfs","nfs",i) for i in dat["Des"]]
  dat["Des"]=[re.sub(r"[1|l][o|0]an","loan",i) for i in dat["Des"]]
  dat["Des"]=[re.sub("[\n]|[\r]"," ",i) for i in dat["Des"]]


  dat["Des"]=[re.sub(r"\(|\)|\[|\]"," ",i) for i in dat["Des"]]#brackets

  dat["Des"]=[re.sub("-|:|\.|/"," ",i) for i in dat["Des"]]


  dat["Des"]=[re.sub("a c ","ac ", i) for i in dat["Des"]]




  dat["Des_cl"]= [re.sub(" ","",i) for i in dat["Des"]]

  #########################################################################
  "Classification"



  dat["classification"]="Not_Tagged"
  # si as transfer
  dat["classification"]=["transfer" if len(re.findall(r"\bsi\b",x))>0 else y for x,y in zip(dat["Des"], dat["classification"])]
  dat["classification"]=["transfer" if len(re.findall(r"\bs i\b",x))>0 else y for x,y in zip(dat["Des"], dat["classification"])]
  
  dat["classification"]=["dd" if len(re.findall(r"\bdd\b",x))>0 else y for x,y in zip(dat["Des"], dat["classification"])]
  
  # ib also as transfer
  dat["classification"]=["ib" if len(re.findall(r"\bib\b",x))>0 else y for x,y in zip(dat["Des"], dat["classification"])]
  
  #ft as transfer 
  dat["classification"]=["transfer" if len(re.findall(r"\bft\b",x))>0 else y for x,y in zip(dat["Des"], dat["classification"])]


  dat["classification"]=["brought_fwd" if len(re.findall("brought_fwd",x))>0 else y for x,y in zip(dat["Des_cl"], dat["classification"])]




  dat["classification"]=["transfer" if len(re.findall(r"\bdr\b",x))>0 else y for x,y in zip(dat["Des"], dat["classification"])]
  dat["classification"]=["transfer" if len(re.findall(r"\bftr\b",x))>0 else y for x,y in zip(dat["Des"], dat["classification"])]
  dat["classification"]=["transfer" if len(re.findall(r"\btpf[t|r]\b",x))>0 else y for x,y in zip(dat["Des"], dat["classification"])]
  dat["classification"]=["transfer" if len(re.findall(r"\bfund[s]? trf\b",x))>0 else y for x,y in zip(dat["Des"], dat["classification"])]
  dat["classification"]=["transfer" if len(re.findall(r"\bmob[\s]?tpft\b",x))>0 else y for x,y in zip(dat["Des"], dat["classification"])]  
  dat["classification"]=["transfer" if len(re.findall(r"\btrf\b",x))>0 else y for x,y in zip(dat["Des"], dat["classification"])]    
  dat["classification"]=["transfer" if len(re.findall("tpt",x))>0 else y for x,y in zip(dat["Des_cl"], dat["classification"])]
  dat["classification"]=["transfer" if len(re.findall("transfer",x))>0 else y for x,y in zip(dat["Des_cl"], dat["classification"])]

  
  
  dat["classification"]=["neft" if len(re.findall("neft",x))>0 else y for x,y in zip(dat["Des_cl"], dat["classification"])]

  dat["classification"]=["rtgs" if len(re.findall("rtgs",x))>0 else y for x,y in zip(dat["Des_cl"], dat["classification"])]


  dat["classification"]=["imps" if len(re.findall(r"[i|1]mps",x))>0 else y for x,y in zip(dat["Des_cl"], dat["classification"])]

  dat["classification"]=["cash" if len(re.findall("cash",x))>0 else y for x,y in zip(dat["Des"], dat["classification"])]

  dat["classification"]=["cash" if len(re.findall("at[m|w]",x))>0 else y for x,y in zip(dat["Des_cl"], dat["classification"])]

  dat["classification"]=["cash" if len(re.findall("self",x))>0 else y for x,y in zip(dat["Des_cl"], dat["classification"])]

  dat["classification"]=["cash" if len(re.findall(r"\bpos\b",x))>0 else y for x,y in zip(dat["Des"], dat["classification"])]

  dat["classification"]=["cash" if len(re.findall("debitcard",x))>0 else y for x,y in zip(dat["Des_cl"], dat["classification"])]

  dat["classification"]=["cheque" if len(re.findall(r"che?q",x))>0 else y for x,y in zip(dat["Des"], dat["classification"])]

  dat["classification"]=["cheque" if len(re.findall("c[l|1][g|q]",x))>0 else y for x,y in zip(dat["Des"], dat["classification"])]

  dat["classification"]=["cheque" if len(re.findall(r"c[l|1]earing",x))>0 else y for x,y in zip(dat["Des_cl"], dat["classification"])]

  
  #cheque for INF
  dat["classification"]=["cheque" if len(re.findall(r"\b[i|1]nf\b",x))>0 else y for x,y in zip(dat["Des"], dat["classification"])]
  
  #cash for mmt
  dat["classification"]=["cash" if len(re.findall(r"\bmmt\b",x))>0 else y for x,y in zip(dat["Des"], dat["classification"])]
  
  
  #ecs, nach, emi, loan are all nach/emi
  dat["classification"]=["nach/emi" if len(re.findall("ecs",x))>0 else y for x,y in zip(dat["Des_cl"], dat["classification"])]

  dat["classification"]=["nach/emi" if len(re.findall("loan",x))>0 else y for x,y in zip(dat["Des_cl"], dat["classification"])]

  dat["classification"]=["nach/emi" if len(re.findall(r"em[i|1|u]",x))>0 else y for x,y in zip(dat["Des_cl"], dat["classification"])]

  dat["classification"]=["nach/emi" if len(re.findall("nach",x))>0 else y for x,y in zip(dat["Des_cl"], dat["classification"])]

  dat["classification"]=["nach/emi" if len(re.findall(r"\bach\b",x))>0 else y for x,y in zip(dat["Des"], dat["classification"])]

  
  dat["classification"]=["i/w" if len(re.findall("inward",x))>0 else y for x,y in zip(dat["Des_cl"], dat["classification"])]
  dat["classification"]=["o/w" if len(re.findall("outward",x))>0 else y for x,y in zip(dat["Des_cl"], dat["classification"])]
  
  dat["classification"]=["i/w" if len(re.findall(r"iwclg",x))>0 else y for x,y in zip(dat["Des_cl"], dat["classification"])]
  dat["classification"]=["o/w" if len(re.findall(r"owclg",x))>0 else y for x,y in zip(dat["Des_cl"], dat["classification"])]

  #see if we need owclg as a clg or a ow 
  #dat["classification"]=["i/w" if len(re.findall(r"\biw\b",x))>0 else y for x,y in zip(dat["Des_cl"], dat["classification"])]
  #dat["classification"]=["o/w" if len(re.findall(r"\bow\b",x))>0 else y for x,y in zip(dat["Des_cl"], dat["classification"])]
  
  
  dat["classification"]=["int_coll" if len(re.findall("int coll",x))>0 else y for x,y in zip(dat["Des"], dat["classification"])]
  
  dat["classification"]=["internet_banking" if len(re.findall(r"\bup[i|1]",x))>0 else y for x,y in zip(dat["Des_cl"], dat["classification"])]
  
  dat["classification"]=["tax" if len(re.findall(r"tax",x))>0 else y for x,y in zip(dat["Des"], dat["classification"])]
  dat["classification"]=["tax" if len(re.findall(r"\btds\b",x))>0 else y for x,y in zip(dat["Des"], dat["classification"])]
  
  dat["classification"]=["gst" if len(re.findall(r"[s|c]gst|\bgst",x))>0 else y for x,y in zip(dat["Des"], dat["classification"])]
  #charges!
  dat["cl_cl"]="Not_Tagged"
  dat["cl_cl"]=["charges" if len(re.findall(r"charge?",x))>0 else y for x,y in zip(dat["Des_cl"], dat["cl_cl"])]
  dat["cl_cl"]=["charges" if len(re.findall("chrg",x))>0 else y for x,y in zip(dat["Des_cl"], dat["cl_cl"])]
  dat["cl_cl"]=["charges" if len(re.findall(r"\bchgs?\b",x))>0 else y for x,y in zip(dat["Des_cl"], dat["cl_cl"])]
  dat["cl_cl"]=["charges" if len(re.findall("commission",x))>0 else y for x,y in zip(dat["Des_cl"], dat["cl_cl"])]
  dat["cl_cl"]=["charges" if len(re.findall(r"\bfee\b",x))>0 else y for x,y in zip(dat["Des"], dat["cl_cl"])]
  dat["cl_cl"]=["charges" if len(re.findall(r"\bnftchg\b",x))>0 else y for x,y in zip(dat["Des"], dat["cl_cl"])]
  dat["cl_cl"]=["charges" if len(re.findall(r"\bchg[s|\s]\b",x))>0 else y for x,y in zip(dat["Des"], dat["cl_cl"])]
  #return
  #dat["cl_ret"]="Not_Tagged"
  #dat["cl_cl"]=["return" if len(re.findall(r"\bretu?r?n?|return",x))>0 else y for x,y in zip(dat["Des"], dat["cl_cl"])]
  dat["cl_cl"]=["return" if len(re.findall(r"\bretu?r?n?\b|return",x))>0 else y for x,y in zip(dat["Des"], dat["cl_cl"])]
  dat["cl_cl"]=["return" if len(re.findall(r"\brtn\b",x))>0 else y for x,y in zip(dat["Des"], dat["cl_cl"])]
  #dat["gst"]="Not_Tagged"
  #dat["classification"]=["gst" if len(re.findall(r"[s|c]gst|\bgst",x))>0 else y for x,y in zip(dat["Des"], dat["gst"])]
  dat["Des"]=["".join([j for j in i if j.isdigit()==False ]) for i in fdf["Des"]]



  #dat["Des"]=[" ".join([j for j in i.split() if not any(c.isdigit() for c in j)]) for i in dat["Des"]]
  #if any alpha numerics are still left

  dat["Des"]= [re.sub("[\W_]+"," ",i) for i in dat["Des"]]
  print("time taken:",time()-t0, "seconds")
  return(dat)




#fdf=clean_transc(fdf)

def pattern(dat):
#  from fuzzywuzzy import fuzz
#  from fuzzywuzzy import process

  
  def sortSecond(val): 
      return val[1]  

  
  dfg=pd.DataFrame( columns =['Debits', 'Credits']) 
  
  narrations = list(dat["Des"][dat["Credit"]==0])
  #jk=narrations
  #remove stopwords, to improve results
  stopwords= ["neft", "rtgs","imps","cash", "cheque", "tpt", "transfer", "to", "atm","dd","clg","atw","self"]
  narrations=[" ".join([j for j in i.split() if j not in stopwords]) for i in narrations]
  list_groups_debit = []
  group_count = 0

  for i,string in enumerate(narrations):
    print(i)
   #if group_count>1000:
   #  break
   #else:
    #print(i,string)
    try:
    
      match_list = process.extract(string, narrations, scorer = fuzz.token_set_ratio, limit = len(narrations))     
      match_list = [ele[0] for ele in match_list if ele[1] >80]

      if len(match_list) > 5: #in list(range(2, 10)):
          list_groups_debit.append(tuple((match_list, len(match_list), group_count)))
          print(group_count)
          group_count +=1


      #for ele in match_list:
      #  narrations.remove(ele)
      narrations=[i for i in narrations if i not in match_list]
        #print("a")
      #print("cccccccccccccc")
    except:
      print("skipping",i,string)

 
  list_groups_debit.sort(key = sortSecond,reverse = True)
  

  dfg["Debits"]=pd.Series([(i[0][1],i[1]) for i in list_groups_debit[0:10]])
  #dfg["Debits-Counts"]=pd.Series([i[1] for i in list_groups_debit[0:10]])



  narrations = list(dat["Des"][dat["Debit"]==0])
  list_groups_credit = []
  group_count = 0

  
  for i,string in enumerate(narrations):
    try:
    #print(i,string)
      match_list = process.extract(string, narrations, scorer = fuzz.token_set_ratio, limit = len(narrations))     
      match_list = [ele[0] for ele in match_list if ele[1] >80]
      if len(match_list) > 5: #in list(range(2, 10)):
          list_groups_credit.append(tuple((match_list, len(match_list), group_count)))
          print(group_count)
          group_count +=1


      #for ele in match_list:
      #  narrations.remove(ele)
      narrations=[i for i in narrations if i not in match_list]
      #print("a")
      #print("cccccccccccccc")
    except:
      print("skipping",i,string)

#  def sortSecond(val): 
#      return val[1]  

  list_groups_credit.sort(key = sortSecond,reverse = True)

  #dfg["Credits"]=[i[0][1] for i in list_groups_credit[1:10]]
  dfg["Credits"]=pd.Series([(i[0][1],i[1]) for i in list_groups_credit[0:10]])
  #gh=[i[0][1] for i in list_groups_credit[1:10]]
  #[y if y else i for i,y in zip(dfg["Credits"],gh)]
  
  return(dfg)
  




def round_entry(dat):
 
 dat["roundiw"]=0
 dat["roundow"]=0
 print("Finding round entries...")

 for i in range(0,len(dat)):
   #print(i,"of", len(dat))
   for j in range(i,i+5):
#    print(j)
     try:
         #debit and then credit
        
         if dat.Debit.iloc[i]==dat.Credit.iloc[j] and dat.Debit.iloc[i]!=0 :
           #print("check........ {}".format(i))
           dat['roundiw'].iloc[i]="inward debit suspected {}".format(i)
           dat['roundiw'].iloc[j]="inward credit suspected {}".format(i)
           if dat["Date_new"].iloc[i]==dat["Date_new"].iloc[j]: #for bounce
             dat['roundiw'].iloc[j]="inward credit suspected same date {}".format(i)

         if dat.Credit.iloc[i]==dat.Debit.iloc[j] and dat.Credit.iloc[i]!=0 :
           #print("check........ {}".format(i))
           dat['roundow'].iloc[i]="outward credit suspected {}".format(i)
           dat['roundow'].iloc[j]="outward debit suspected {}".format(i)
           if dat["Date_new"].iloc[i]==dat["Date_new"].iloc[j]: #for bounce
             dat['roundow'].iloc[j]="outward debit suspected same date{}".format(i)

            
     except:
       print("NA")
 print(dat['roundiw'].value_counts())
 print(dat['roundow'].value_counts())
   
 return(dat)   







fdf= pd.read_csv("df.csv")
dat=fdf[fdf.counter==51]
dat=dat.drop(["round","rt","Des","classification","cl_cl","gst"], axis=1)
#from sklearn.pipeline import Pipeline

#trns=  Pipeline([("clean",clean_transc(dat)),("pat",pattern(dat)),("round",round_entry(dat))])

#trns=  Pipeline([])
#needs class not def


list(dat)
dat= clean_transc(dat)
dat= round_entry(dat)
pat=pattern(dat)


fdf["iw_see"]=""
fdf["Description"]=[str(i).lower() for i in fdf["Description"]]
fdf["iw_see"]=["iw" if len(re.findall(r"iw",x))>0 else y for x,y in zip(fdf["Description"], fdf["iw_see"])]
fdf["iw_see"].value_counts()

#pattern---->r"\biw\b"

fdf["ow_see"]=""
fdf["Description"]=[str(i).lower() for i in fdf["Description"]]
fdf["ow_see"]=["ow" if len(re.findall(r"ow",x))>0 else y for x,y in zip(fdf["Description"], fdf["ow_see"])]
fdf["ow_see"].value_counts()
#pattern--> r"\bow\b"


###############################################################################
###############################################################################





#import datetime
from sklearn import preprocessing
import matplotlib.pyplot as plt

def plot_months(dat):
  
  dat['DateTime'] = pd.to_datetime(dat['Date_new'])
  #dat.DateTime.iloc[1].day
  dat["day"]=dat.DateTime.apply(lambda x: x.day)
  dat["month"]=dat.DateTime.apply(lambda x: x.month)
  dat["year"]=dat.DateTime.apply(lambda x: x.year)

  dat=dat[["day","month","year", "Balance"]]


  df=pd.DataFrame()
  mw=pd.DataFrame({'day': range(1,32)})

  for i in set(dat["year"]):
    dat1=dat[dat.year==i]
    for j in set(dat["month"]):
      dat2= dat1[dat1.month==j]
      #df.sort_values('date').groupby('id').tail(1)
      dat3=dat2.groupby('day').tail(1)
      dat4=pd.merge(mw,dat3,on='day', how='left')
      dat4.year=i
      dat4.month=j
      dat4=dat4.fillna(0)
      #cant fill the 0 in bal now as we will need the values in initial days from the past month, if they are 0.    
      df=df.append(dat4)

#df["bal_cum"]=df.bal.cumsum()
  print("Balance data takes a lil time...")
  for k in range(1,len(df)):
    #print("Balance data takes a lil time...")
    print('.', end='', flush=True)
    if df.Balance.iloc[k]==0:
      df.Balance.iloc[k]=df.Balance.iloc[k-1]
  
  """df["bul"]= [i if i!=0 else i.shift(-1) for i in dat.Balance[1:]]"""
    
#df["bal_cum"]= [i-j for i,j in zip(df["Credit"], df["Debit"])]

#df["bal_cum_norm2"]=preprocessing.scale(df.bal_cum)


  for year in set(df.year):
    print(year)
    dfy=df[df.year==year]
    # plot data
    fig, ax = plt.subplots(figsize=(15,7))
    # use unstack()
    dfy.groupby(['day','month']).sum()['Balance'].unstack().plot(ax=ax) 
  
    plt.savefig('Month_day_Balance_{}.png'.format(year))


plot_months(fdf)

###############################################################################
###############################################################################
#Variance
def variance(dat):
  dat["cum"]= dat.Balance.cumsum()
  dfv=dat[dat["cum"]!=0]
  for year in set(dfv.year):
    dfy=dfv[dfv.year==year]
    plt.plot(preprocessing.scale(dfy.groupby(['day','month']).sum()['Balance'].unstack().var()))
    plt.title('Balance Variance across Months')
    plt.ylabel('Variance')
    plt.xlabel('Months')
    plt.legend(list(set(dat.year)), loc='upper right')
    plt.savefig('Variance_Balance.png'.format(year))

variance(dat)


dat=fdf[fdf.counter==39]






