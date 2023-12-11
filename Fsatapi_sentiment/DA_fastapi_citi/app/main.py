# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 18:53:24 2022

@author: ELECTROBOT
"""

from fastapi import FastAPI, Depends
from app.routers import  upload, search
import os



app= FastAPI(root_path=os.environ.get("SCRIPT_NAME"))
# Define a directory to store uploaded files
upload_dir = "uploads"
os.makedirs(upload_dir, exist_ok=True)  # Create the directory if it doesn't exist



app.include_router(upload.router)
app.include_router(search.router)


@app.get("/")
async def root():
    return {"message":"Hello World"}


