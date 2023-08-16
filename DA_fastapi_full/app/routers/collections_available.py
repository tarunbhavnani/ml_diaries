from fastapi import APIRouter, Depends, Request, UploadFile, File
from app.dependencies import fetch_user
from typing import Union, List
from app.functions import get_redis_keys
from app.logger import logger
from app.response_def import ListResponse


router= APIRouter(
    prefix="/collections_available",
    tags= ["collections_available"],
    dependencies=[],
    responses={404:{"description":"Not Found"}},

)


@router.get("/", response_model= ListResponse)
def collections_available(user= Depends(fetch_user))->ListResponse:
    names= get_redis_keys(collection="all_collections_fp", userr_id= str(user.id))
    return ListResponse(names=names)


