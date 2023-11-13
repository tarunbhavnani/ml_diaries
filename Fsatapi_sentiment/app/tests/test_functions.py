import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
from app.main import app
import torch
from app.functions import get_sentiment
client = TestClient(app)



#def test_get_sentiment():
#    get_sentiment = Mock(return_value="mocked stuff")
#
#    resp= get_sentiment()
#    assert resp=="mocked stuff"


#pytest app/tests -v --cov-report term --cov-report html:htmlcov --cov-report xml --cov-fail-under=90



def test_get_sentiment():
    # create a mock object for the model and tokenizer
    model_mock = Mock()
    tokenizer_mock = Mock()

    # define the expected return value for the model and tokenizer mocks
    model_mock.return_value = [torch.tensor([[0.5, 0.3, 0.2]])]
    tokenizer_mock.return_value = {"input_ids": torch.tensor([[1, 2, 3]]),
                                   "attention_mask": torch.tensor([[1, 1, 1]])}

    # create a monkeypatch object to replace the model and tokenizer with the mock objects
    with patch("app.functions.get_sentiment","model", model_mock), patch("app.functions.get_sentiment","tokenizer", tokenizer_mock):
        # call the get_sentiment function with the mock objects
        result = get_sentiment("some text", model_mock, tokenizer_mock)

        # assert that the result is as expected
    assert result["negative"]>0
