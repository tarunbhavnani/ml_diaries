# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 18:53:24 2022

@author: ELECTROBOT
"""

from fastapi import FastAPI, Depends
from app.routers import sentiment
import os



app= FastAPI(root_path=os.environ.get("SCRIPT_NAME"))

app.include_router(sentiment.router)


@app.get("/")
async def root():
    return {"message":"Hello World"}


