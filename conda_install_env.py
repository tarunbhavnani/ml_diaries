#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 13:44:27 2019

@author: tarun.bhavnani
"""


conda create -n -p /home/tarun.bhavnani/anaconda3/envs/tarun python=3.6


conda activate /home/tarun.bhavnani/anaconda3/envs/tarun


spyder --new-instance

tools--> preferences

python interpreter--> use following interpreter
/home/tarun.bhavnani/anaconda3/envs/tarun/bin/python


#pip install tensorflow==1.4.0
#pip install --upgrade keras==2.1.3 # TypeError: softmax() got an unexpected keyword argument 'axis'


pip install tensorflow==1.5.0


#for multiple ipython consoles on spyder
conda install ipykernel cloudpickle



