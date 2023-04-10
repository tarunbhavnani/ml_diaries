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

#not avalilable -->wsecondary=pd.read_csv("WSecondaryTourneyCompactResults.csv")

sample_sub= pd.read_csv("SampleSubmissionWarmup.csv")


mteamconf= pd.read_csv("MTeamConferences.csv")
wteamconf= pd.read_csv("WTeamConferences.csv")
#['Season', 'TeamID', 'ConfAbbrev']

mncaadetails=pd.read_csv( 'MNCAATourneyDetailedResults.csv')
wncaadetails=pd.read_csv( 'WNCAATourneyDetailedResults.csv')
#['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 
#'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']


mdetails=pd.read_csv( 'MRegularSeasonDetailedResults.csv')
wdetails=pd.read_csv( 'WRegularSeasonDetailedResults.csv')
#['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 
#'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']





mgamecity= pd.read_csv("MGameCities.csv")
wgamecity= pd.read_csv("WGameCities.csv")


mrankings=pd.read_csv("MMasseyOrdinals.csv")
hd=mrankings.head(1000)


mcoaches= pd.read_csv("MTeamCoaches.csv")


mconfgames=pd.read_csv("MConferenceTourneyGames.csv")

mstt= pd.read_csv("MSecondaryTourneyTeams.csv")
mstt_results= pd.read_csv("MSecondaryTourneyCompactResults.csv")


mteam_spelling= pd.read_csv("MTeamSpellings.csv",  encoding = 'cp1252')
wteam_spelling= pd.read_csv("WTeamSpellings.csv",encoding = 'cp1252')


mncaa_slots=pd.read_csv("MNCAATourneySlots.csv")
wncaa_slots=pd.read_csv("WNCAATourneySlots.csv")


mncaa_sead_slots=pd.read_csv("MNCAATourneySeedRoundSlots.csv")





# =============================================================================
# check if we hae the seed informations at the time of prediction

# do we do a prediction of for example goals scored against teams, and then do a prediction for all the parameters in details.
# then from the parameters predicted , predict who will win
# the who will model will be the details against win/loss, something like that!


# =============================================================================

#add seed rankings to ncss teams!

wncaadetails= wncaadetails.merge(wncaa.rename(columns={'Seed': 'WSeed'}), right_on= ['Season', 'TeamID'], left_on= ['Season', 'WTeamID'])
wncaadetails.drop("TeamID", axis=1, inplace=True)
wncaadetails= wncaadetails.merge(wncaa.rename(columns={'Seed': 'LSeed'}), right_on= ['Season', 'TeamID'], left_on= ['Season', 'LTeamID'])
wncaadetails.drop("TeamID", axis=1, inplace=True)


mncaadetails= mncaadetails.merge(mncaa.rename(columns={'Seed': 'WSeed'}), right_on= ['Season', 'TeamID'], left_on= ['Season', 'WTeamID'])
mncaadetails.drop("TeamID", axis=1, inplace=True)
mncaadetails= mncaadetails.merge(mncaa.rename(columns={'Seed': 'LSeed'}), right_on= ['Season', 'TeamID'], left_on= ['Season', 'LTeamID'])
mncaadetails.drop("TeamID", axis=1, inplace=True)



mncaadetails= mncaadetails.merge(mncaa.rename(columns={'Seed': 'WSeed'}), right_on= ['Season', 'TeamID'], left_on= ['Season', 'WTeamID'])
mncaadetails.drop("TeamID", axis=1, inplace=True)
mncaadetails= mncaadetails.merge(mncaa.rename(columns={'Seed': 'LSeed'}), right_on= ['Season', 'TeamID'], left_on= ['Season', 'LTeamID'])
mncaadetails.drop("TeamID", axis=1, inplace=True)



# =============================================================================









