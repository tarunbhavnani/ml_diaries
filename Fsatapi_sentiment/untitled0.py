# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 18:43:35 2022

@author: ELECTROBOT
"""
from typing import Union
from fastapi import APIRouter, Depends, HTTPException, Request
from functions import get_sentiment
from pydantic import BaseModel, Field




Model_path=""
model=
tokenizer=



router= APIRouter(
    prefix="/sentiment",
    tags=["sentiment"],
    dependencies=[],
    responses={404: {"descfription":"Not found"}},
    )


class Item(BaseModel):
    text:Union(str, None) = Field(
        default=None, title="text blob", max_lengtmax_length)


@router.post('/')
async def sentiment(item:Item):
    """
    

    Parameters
    ----------
    item : Item
        DESCRIPTION.

    Returns
    -------
    None.

    """
    try:
        sentiment_result= get_sentiment(item.text, model, tokenizer)
        sentiment_result= {"negative":sentiment_result["Negative"],
                           "neutral":sentiment_result["Neutral"],
                           "positive":sentiment_result["Positive"]}
        response= {"text": item.text, "sentimentResult": sentiment_result}
        
        return response
    except Exception as e:
        raise HTTPException(status_code=404, details=str(e))
        