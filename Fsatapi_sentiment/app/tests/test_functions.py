import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
from app.main import app
from app.functions import get_sentiment
client = TestClient(app)



def test_get_sentiment():
    get_sentiment = Mock(return_value="mocked stuff")

    resp= get_sentiment()
    assert resp=="mocked stuff"


#pytest app/tests -v --cov-report term --cov-report html:htmlcov --cov-report xml --cov-fail-under=90




