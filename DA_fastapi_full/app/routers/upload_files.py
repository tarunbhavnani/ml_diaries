from fastapi import APIRouter, Depends, Request, UploadFile, File
from app.dependencies import fetch_user
from typing import Union, List
from app.functions import process_upload_files
from app.logger import logger
from app.response_def import ListResponse
from fastapi.responses import Response, FileResponse


router= APIRouter(
    prefix="/upload_files",
    tags= ["upload_files"],
    dependencies=[],
    responses={404:{"description":"Not Found"}},

)


@router.get("/{collection}", response_model= ListResponse)
def upload_files(collection=None, user= Depends(fetch_user), files: List[UploadFile]=File(...))->ListResponse:
    names= process_upload_files(files=files, collection=collection, user_id=str(user.id))
    return ListResponse(names=names)


