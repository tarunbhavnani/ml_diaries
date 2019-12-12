#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 14:19:56 2019

@author: tarun.bhavnani
"""

from flask import Flask

app= Flask(__name__)

@app.route("/")
def hello():
    return "helloooooooooooooooooooooooooooo"


if __name__=='__main__':
    app.run(debug=True)
    
