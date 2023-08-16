from fastapi import APIRouter, Depends, Request, UploadFile, File
from app.dependencies import fetch_user
from typing import Union, List
from app.functions import delete_files
from app.logger import logger
from app.response_def import ListResponse
from fastapi.responses import Response, FileResponse


router= APIRouter(
    prefix="/reset",
    tags= ["reset"],
    dependencies=[],
    responses={404:{"description":"Not Found"}},

)


@router.get("/{collection}", response_model= ListResponse)
def reset(collection=None, user= Depends(fetch_user))->ListResponse:
    names= delete_files(collection=collection, user_id= str(user.id))

    return ListResponse(names=names)



