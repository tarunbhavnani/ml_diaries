#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 10:37:42 2019

@author: tarun.bhavnani@dev.smecorner.com

one counter for last uttered, in case of repeat the bot still is on the last utterance
one counter for the previous action as well in case of rep


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import requests
import json
from rasa_core_sdk import Action
from rasa_core_sdk.events import SlotSet
from rasa_core_sdk.forms import FormAction
#from rasa_core.events import UserUtteranceReverted
from rasa_core_sdk.events import UserUtteranceReverted
from rasa_core_sdk.events import ActionReverted
from rasa_core_sdk.events import FollowupAction
from rasa_core.interpreter import RasaNLUInterpreter
#import pandas as pd
#import xlrd
import re
from word2number import w2n
from word2number import w2n
import nltk
import spacy
import datetime
import requests
nlp= spacy.load("en")

#df = pd.read_excel('data_los.xlsx')

logger = logging.getLogger(__name__)



class ActionDefaultFallback(Action):
    def name(self):
        #return "action_question_counter"
        return 'action_default_fallback'
    def run(self, dispatcher, tracker, domain):
        counter= tracker.get_slot('counter')
        current=tracker.get_slot('current')
        bkind= tracker.get_slot("bkind")
        nob= tracker.get_slot("nob")
        industry=tracker.get_slot("industry")
        interview_state= tracker.get_slot("interview_state")
        last_intent= tracker.latest_message['intent'].get('name')
        
        last_message= tracker.latest_message['text']
        last_message= last_message.lower()
        user_name= tracker.get_slot("user_name")
        digits=[i for i in re.findall('\d+', last_message )]
        date= next(tracker.get_latest_entity_values("DATE"), None)
        cardinal= next(tracker.get_latest_entity_values("CARDINAL"), None)
        #dispatcher.utter_message(current)
        #dispatcher.utter_message(counter)
        #dispatcher.utter_message(last_intent)
        #json_response= tracker.get_slot("json_response")
        #dispatcher.utter_message(json_response)
        
        #before interview start
        #interview state turns to "started if details are fetched in action fetch details"
        
        if interview_state == "start":
          if last_intent=="greet":         
            dispatcher.utter_message("अब हम पीडी शुरू करने के लिए आगे बढ़ेंगे। यदि आप किसी भी समय इंटरव्यू से बाहर निकलना चाहते हैं, तो 'stop' इनपुट करें.")
            counter="action_interview_start"
            return[FollowupAction(counter)]
          elif counter !="action_interview_start":
            
            dispatcher.utter_message("शुरू करने के लिए कृपया 'Hi' इनपुट करें.")
            counter="action_listen"
            return[FollowupAction(counter)]

        
        #interview starter
        if current==counter=="action_interview_start":
          counter="action_fetch_details"
          return[FollowupAction(counter)]
        
        
        #after interview start
        if interview_state=="started":
          
          
          #check for intents that are out of scope for interview!
          #in these we followup with the same question
          
          #blank reply
          
          if len(last_message)==0:
            dispatcher.utter_message("कृपया उत्तर को खाली न छोड़ें, मैं फिर पूछूंगा.")
            return[FollowupAction(current)]
          
          #repeat
          
          if (last_intent=="repeat") or (last_message=="what"):
            dispatcher.utter_message("मैं फिर पूछूंगा.")           
            return[FollowupAction(current)]
          
          #chitchat
          
          if last_intent=="chitchat":
            #dispatcher.utter_message("This is an interview, ill ask again!!")
            dispatcher.utter_template("utter_chitchat", tracker)
            return[FollowupAction(current)]
          
          
          #greet
          
          if last_intent=="greet":
            #dispatcher.utter_message("This is an interview, ill ask again!!")
            dispatcher.utter_template("utter_greet", tracker)
            return[FollowupAction(current)]        

          
          #thank
          
          if last_intent=="thank":
            dispatcher.utter_template("utter_thanks", tracker)
            return[FollowupAction(current)]
        
        
          #goodbye
          
          if last_intent=="goodbye":
            #dispatcher.utter_message("Goodbye {}".format(user_name))
            #counter="action_stop_check"
            return[FollowupAction("action_stop_check")]
        

          #staring the interview with 12345

          if current=="action_fetch_details":
            user_name= last_message
            return[FollowupAction(counter),SlotSet('user_name', user_name) ]  
        
          #stop
          
          if (last_message=="stop") or (last_intent=="stop"):
            #dispatcher.utter_message("Goodbye {}".format(user_name))
            #counter="action_stop_check"
            return[FollowupAction("action_stop_check")]
  
          
          if current=="action_stop_check":
            if last_intent== "affirm":
              counter= "action_stop"
            else:
              dispatcher.utter_message("हम PD जारी रखेंगे।")
              #will use the same current and ask again.
              #note thatw e have put the current in counter in action_stop_check for a recall.
            return[FollowupAction(counter)]  

          
          #business kind
          
          if current=="action_business_kind":
            if (last_intent=="pvt") or (last_message=="private"):
               #dispatcher.utter_message("Got it u meant private!")
               counter="action_private"
               bkind="private"
            elif (last_intent== "public") or (last_message=="public"):
               #dispatcher.utter_message("Got it u meant public!")
               counter= "action_public"
               bkind="public"
            elif (last_intent=="prop") or (last_message=="prop"):
             #dispatcher.utter_message("Got it u meant proprietery!")
               counter="action_business_years"
               bkind="prop"
            elif (last_intent== "partnership") or (last_message=="partnership"):
             #dispatcher.utter_message("Got it u meant partnership!")
               counter="action_partner"
               bkind="partnership"
            else:
               dispatcher.utter_message("Not understood!")
               counter="action_business_kind"
            return[FollowupAction(counter),SlotSet('bkind', bkind)]
         
          
          
          #Nature of business!!
          
          if current == "action_nob":
             
             if (last_intent=="manufacturing") or (last_message=="manufacturing"):
               #dispatcher.utter_message("Manufacturing!")
               nob="manu"
               #counter= "action_industry_followup"
               return[FollowupAction(counter),SlotSet('nob', nob)]
               #counter= "action_manu_loc"
             elif (last_intent=="SP") or (last_message=="SP"):
               dispatcher.utter_message("सर्विस प्रोवाइडर!")
               nob="sp"
               #counter= "action_industry_followup"
               return[FollowupAction(counter),SlotSet('nob', nob)]
               #counter= "action_sp_order"
             elif (last_intent== "trader") or (last_message=="trader"):
               dispatcher.utter_message("Trader!")
               nob="trader"
               #counter= "action_industry_followup"
               return[FollowupAction(counter),SlotSet('nob', nob)]
               #counter= "action_trader"
             else:
               dispatcher.utter_message("कृपया जवाब दें!")
               dispatcher.utter_template("utter_ask_nob", tracker)
               return[FollowupAction("action_listen")]


            
                   
            
           
          if counter=="end":
               dispatcher.utter_message("आपके समय के लिए धन्यवाद!")
               return[FollowupAction("action_stop")]
        
        
        return[FollowupAction(counter)]
        

