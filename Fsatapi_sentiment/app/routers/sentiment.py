# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 18:43:35 2022

@author: ELECTROBOT
"""
from typing import Union
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from app.functions import get_sentiment
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
Model_path = r"C:\Users\ELECTROBOT\Desktop\model_dump\sentiment_roberta"

tokenizer = AutoTokenizer.from_pretrained(Model_path)
# config = AutoConfig.from_pretrained(Model_path)
# PT
model = AutoModelForSequenceClassification.from_pretrained(Model_path)


router= APIRouter(
    prefix="/sentiment",
    tags=["sentiment"],
    dependencies=[],
    responses={404: {"descfription":"Not found"}},
    )


class Item(BaseModel):
    text:Union[str, None] = Field(
        default=None, title="text blob",max_length=3000)



@router.post('/')
def get_sentiment_scores(item:Item):
    sentiment_result= get_sentiment(item.text, model, tokenizer)

    response= {"text": item.text, "sentimentResult": sentiment_result}

    return response