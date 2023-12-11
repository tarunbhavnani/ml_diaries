# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 18:43:35 2022

@author: ELECTROBOT
"""
from typing import Union
from fastapi import APIRouter, Depends, Request, UploadFile, File,HTTPException

from pydantic import BaseModel, Field
from app.functions import process_upload_files
from typing import List, Union


router= APIRouter(
    prefix="/upload_files",
    tags= ["upload_files"],
    dependencies=[],
    responses={404:{"description":"Not Found"}},

)


class ListResponse(BaseModel):
    names: List[str]

@router.post("/", response_model= ListResponse)
def upload_files( files: List[UploadFile]=File(...))->ListResponse:
    names= process_upload_files(files=files)
    return ListResponse(names=names)


# @router.post("/upload_files/")
# async def upload_files(files: List[UploadFile] = File(...))->ListResponse:
#     try:
#         # Add debugging information
#         for file in files:
#             print("Received file:", file.filename)
#             print("File size:", file.file.read(1024))
#         names = process_upload_files(files=files)
#         return ListResponse(names=names)

#         # Process files (your existing logic)

#         #return {"message": "Files uploaded successfully"}
#     except Exception as e:
#         # Print any exception that occurs
#         print("Exception during file upload:", str(e))
#         raise HTTPException(status_code=500, detail="Internal Server Error")