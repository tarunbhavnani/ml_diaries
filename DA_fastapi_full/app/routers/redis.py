from fastapi import APIRouter, Depends, HTTPException
from typing import Union
from app.dependencies import fetch_user
from app.cache.redis import cache

router= APIRouter(
    prefix=="/redis",
    tags=["redis"],
    dependencies=[],
    responses={404:{'description':"Not Found"}},

)



@router.get('/ping')
def ping_redis(user= Depends(fetch_user)):
    """
    pings redis and check if reachabkle
    """

    return cache.ping()


@router.get('/info')
def ping_redis(server:Union(str,None)=None, user=Depends(fetch_user))
    """
    redis redis server info
    """
    return cache.info(server=server)

