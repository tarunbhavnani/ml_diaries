#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 20:39:01 2019

@author: tarun.bhavnani
"""

#os.chdir('/home/tarun.bhavnani@dev.smecorner.com/Desktop/final_bot/final_bot2')
from rasa_core.channels.slack import SlackInput
from rasa_core.agent import Agent
from rasa_core.interpreter import RasaNLUInterpreter
import yaml
from rasa_core.utils import EndpointConfig
from rasa_core.tracker_store import MongoTrackerStore
import argparse


def agnt(st):
    nlu_interpreter = RasaNLUInterpreter('/app/models/rasa_nlu/current/nlu')

    core='/app/models/rasa_core'

    action_endpoint = EndpointConfig(url="http://localhost:5055/webhook")
    #db= MongoTrackerStore(domain="/app/config/domain.yml",host='mongodb://localhost:27017', db='rasa', username="tarun", 
	#		password="pass123",collection="conversations",event_broker=None)


    #agent = Agent.load(path=core, interpreter = nlu_interpreter, action_endpoint = action_endpoint,tracker_store=db)



    agent = Agent.load(path=core, interpreter = nlu_interpreter, action_endpoint = action_endpoint)

    input_channel = SlackInput(st)
    agent.handle_channels([input_channel], 5005, serve_forever=True)
 

#'xoxb-542065604356-582580159955-see below'
#nuwuIUEQfJQuYGbOWf3MDpZq
    
if __name__ == '__main__':    
     parser = argparse.ArgumentParser()
     parser = argparse.ArgumentParser( description='starts the bot')

     parser.add_argument( 'SLACK')
     SLACK = parser.parse_args().SLACK
     #print(SLACK)
     agnt(st=SLACK)


