# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:56:21 2024

@author: tarun
"""

import os
os.chdir(r'D:\kaggle\MarchML\march-machine-learning-mania-2024')
lst=os.listdir()

import pandas as pd


# Iterate over each file name
missed=[]
for file_name in lst:
    try:
        if file_name.endswith('.csv'):
            # Extract variable name without extension
            var_name = file_name[:-4]
            
            # Read CSV file into DataFrame
            df = pd.read_csv(file_name)
            
            # Assign DataFrame to variable with file name (without extension)
            globals()[var_name] = df
    except:
        missed.append(file_name)



#missed
MTeamSpellings=pd.read_csv("MTeamSpellings.csv",encoding='cp1252')
WTeamSpellings=pd.read_csv("WTeamSpellings.csv",encoding='cp1252')


# =============================================================================
# 
# =============================================================================


check=MRegularSeasonCompactResults[MRegularSeasonCompactResults.Season==2012]

mmasey= MMasseyOrdinals[MMasseyOrdinals.Season==2003]


# kl=MTeamCoaches.groupby('CoachName')['TeamID'].apply(lambda x: set(x)).reset_index()
# kl[[True if len(i)>1 else False for i in kl.TeamID]]
kl0=MNCAATourneySeeds[MNCAATourneySeeds.Season==2023]

kl0= {i:j for i,j in zip(kl0.Seed, kl0.TeamID)}


kl=MNCAATourneySlots[MNCAATourneySlots.Season==2023]
kl['ss']=kl.StrongSeed.map(kl0)
kl['ws']=kl.WeakSeed.map(kl0)



kl1=MNCAATourneyCompactResults[MNCAATourneyCompactResults.Season==2023]


kl1=kl1.merge(kl0, left_on="WTeamID" , right_on='TeamID')
kl1=kl1.merge(kl0, left_on="LTeamID" , right_on='TeamID')
kl1.columns=['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc',
       'NumOT', 'Season_y', 'WSeed', 'TeamID_x', 'Season', 'LSeed',
       'TeamID_y']
kl1=kl1[['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc',
       'NumOT',  'WSeed', 'LSeed']]
