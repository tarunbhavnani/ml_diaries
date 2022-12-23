# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 18:43:35 2022

@author: ELECTROBOT
"""
from fastapi import APIRouter, Depends, HTTPException, Request
#from functions import get_sentiment
from pydantic import BaseModel, Field

from transformers import AutoModelForSequenceClassification
#from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from typing import Union



Model_path=r"C:\Users\ELECTROBOT\Desktop\model_dump\sentiment_roberta"

tokenizer = AutoTokenizer.from_pretrained(Model_path)
#config = AutoConfig.from_pretrained(Model_path)
# PT
model = AutoModelForSequenceClassification.from_pretrained(Model_path)


def softmax(x):
    e_x= np.exp(x-np.max(x))
    return e_x/e_x.sum(axis=0)

def get_sentiment(sent, model, tokenizer):
    encoded_input = tokenizer(sent, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy().astype("float64")
    neg, neutral, pos = softmax(scores)
    
    return {"Negative":neg, "Neutral":neutral,"Positive":pos}






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
        