from fastapi import APIRouter, Depends, Request, UploadFile, File
from app.dependencies import fetch_user
from typing import Union, List
from app.functions import  transfer_fp
from app.logger import logger
from app.response_def import ListResponse


router= APIRouter(
    prefix="/transferfiles",
    tags= ["transferfiles"],
    dependencies=[],
    responses={404:{"description":"Not Found"}},

)


@router.get("/{collection_var}", response_model= ListResponse)
def transfer_files(collection_var= None, user= Depends(fetch_user))->ListResponse:
    names= transfer_fp(collection=collection_var, user_id= str(user.id))
    return ListResponse(names=names)



