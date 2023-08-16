from fastapi import APIRouter, Depends, Request, UploadFile, File
from app.dependencies import fetch_user
from typing import Union, List
from app.functions import get_redis_keys, process_upload_files
from app.logger import logger
from app.response_def import ListResponse


router= APIRouter(
    prefix="/filename_redis",
    tags= ["filename_redis"],
    dependencies=[],
    responses={404:{"description":"Not Found"}},

)


@router.get("/{collection_var}", response_model= ListResponse)
def filename(collection_var= None, user= Depends(fetch_user))->ListResponse:
    logger.info("Recieved filename")
    names= get_redis_keys(collection=collection_var, user_id= str(user.id))
    names= [name for name in names if name[-2:]!="fp"]
    return ListResponse(names)



