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
    prefix="/predict",
    tags=["predict"],
    dependencies=[],
    responses={404: {"descfription":"Not found"}},
    )


qna= None

def load_qna():
    global qna 
    if qna is None:
        qna= Qnatb(model_path=model_path)


@router('/', response_model= ResponseItemPredict1)
def predict(data: RequestBody, user= Depends(fetch_user)) -> ResponsePredict:
    load_qna()
    responses= get_final_response(qna, uestion= data.text)
    response_items= []
    for item in response['results']:
        response_items.append(
            ResponseItemPredict(answer= item["answer"], blob= item['blob']))
    return ResponseItemPredict1(errormsg=responses['errormsg'], result=Responsepredict(responses= response_items), status= responses['status'])

