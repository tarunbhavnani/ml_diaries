from fastapi import APIRouter, Depends, Request, UploadFile, File
from app.dependencies import fetch_user
from typing import Union, List
from app.functions import get_file
from app.logger import logger
from app.response_def import ListResponse
from fastapi.responses import Response, FileResponse


router= APIRouter(
    prefix="/get_pdf",
    tags= ["get_pdf"],
    dependencies=[],
    responses={404:{"description":"Not Found"}},

)


@router.get("/{filename}", response_model= ListResponse)
def get_pdf(filename=None, user= Depends(fetch_user))->ListResponse:
    file_obj= get_file(filename)
    return Response(content=file_obj, media_type= "application/pdf")


