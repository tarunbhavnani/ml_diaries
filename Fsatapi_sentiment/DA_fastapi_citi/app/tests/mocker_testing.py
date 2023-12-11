# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 16:02:51 2023

@author: ELECTROBOT
"""
from unittest.mock import Mock
import torch
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def test_get_sentiment():
    # create a mock object for the model and tokenizer
    model_mock = Mock()
    tokenizer_mock = Mock()

    # define the expected return value for the model and tokenizer mocks
    model_mock.return_value = [torch.tensor([[1, 1, 1]])]
    tokenizer_mock.return_value = {"input_ids": torch.tensor([[1, 1, 0]]),
                                   "attention_mask": torch.tensor([[1, 1, 1]])}

    # create a monkeypatch object to replace the model and tokenizer with the mock objects
    with patch("get_sentiment.model", model_mock), patch("get_sentiment.tokenizer", tokenizer_mock):
        # call the get_sentiment function with the mock objects
        result = get_sentiment("some text", model_mock, tokenizer_mock)

        # assert that the result is as expected
        assert result["negative"]>0
        


def get_sentiment(sent, model, tokenizer):
    encoded_input = tokenizer(sent, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy().astype("float64")
    neg, neutral, pos = softmax(scores)

    return {"negative": neg, "neutral": neutral, "positive": pos}



