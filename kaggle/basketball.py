# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 12:44:06 2023

@author: ELECTROBOT
"""
import pandas as pd
import os
os.chdir("C:\\Users\\ELECTROBOT\\Desktop\\kaggle\\basketball")
os.listdir()


cities= pd.read_csv("Cities.csv")
list(cities)
#['CityID', 'City', 'State']

mteams= pd.read_csv("MTeams.csv")
list(mteams)
#['TeamID', 'TeamName', 'FirstD1Season', 'LastD1Season']
wteams= pd.read_csv("WTeams.csv")
list(wteams)
#['TeamID', 'TeamName']


mseasons= pd.read_csv("MSeasons.csv")
wseasons= pd.read_csv("WSeasons.csv")

#['Season', 'DayZero', 'RegionW', 'RegionX', 'RegionY', 'RegionZ']

#W, X, Y, or Z. Whichever region's name comes first alphabetically, that region will be Region W. And whichever Region plays 
#against Region W in the national semifinals, that will be Region X. For the other two regions, whichever region's name comes first 
#alphabetically, that region will be Region Y, and the other will be Region Z

mncaa= pd.read_csv("MNCAATourneySeeds.csv")
wncaa= pd.read_csv("WNCAATourneySeeds.csv")
#['Season', 'Seed', 'TeamID']


mresults=pd.read_csv("MRegularSeasonCompactResults.csv ")
hd= mresults.head(1000)
wresults=pd.read_csv("WRegularSeasonCompactResults.csv ")
hd= wresults.head(1000)
#['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT']

mncaaresults= pd.read_csv("MNCAATourneyCompactResults.csv")
wncaaresults= pd.read_csv("WNCAATourneyCompactResults.csv")
#['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT']


#secondary mens data
msecondary=pd.read_csv("MSecondaryTourneyCompactResults.csv")
#['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT']


sample_sub= pd.read_csv("SampleSubmissionWarmup.csv")


mteamconf= pd.read_csv("MTeamConferences.csv")
wteamconf= pd.read_csv("WTeamConferences.csv")



mdetails=pd.read_csv( 'MNCAATourneyDetailedResults.csv')
wdetails=pd.read_csv( 'WNCAATourneyDetailedResults.csv')
#['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 
#'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']



# =============================================================================
# check if we hae the seed informations at the time of prediction

# do we do a prediction of for example goals scored against teams, and then do a prediction for all the parameters in details.
# then from the parameters predicted , predict who will win
# the who will model will be the details against win/loss, something like that!


# =============================================================================
