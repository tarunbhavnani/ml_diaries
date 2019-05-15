#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 10:49:29 2019

@author: tarun.bhavnani@dev.smecorner.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import warnings
#from rasa_core.utils import EndpointConfig
import json
logger = logging.getLogger(__name__)
#os.chdir('/home/tarun.bhavnani@dev.smecorner.com/Desktop/latest_bot/latest_bot/interview')

def mul1(x,y):
  return(x*y)
def add1(x,y):
  return(x+y)
def div1(x,y):
  return(x/y)



###run bot!########################################################################
if __name__ == '__main__':
    #utils.configure_colored_logging(loglevel="INFO")

    parser = argparse.ArgumentParser(
            description='starts the bot')

    parser.add_argument(
            'task',
            choices=["mul", "add", "div"],
            help="what the bot should do - e.g. run or train?")
    task = parser.parse_args().task
    x=23
    y=4

    # decide what to do based on first parameter of the script
    if task == "mul":
        print(mul1(x,y))
    elif task == "div":
        div1(x,y)
    elif task =="add":
        add1(x,y)
        
    
#python bot.py train-nlu
#python bot.py train-dialogue
#python bot.py run
#python -m rasa_core.train --online -o models/dialogue -d domain.yml -s data/stories.md --endpoints endpoints.yml
        
        
