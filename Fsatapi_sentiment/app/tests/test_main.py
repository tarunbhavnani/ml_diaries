import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
from app.main import app
from app.functions import get_sentiment
client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}



#pytest app/tests -v --cov-report term --cov-report html:htmlcov --cov-report xml --cov-fail-under=90

#python -m pytest --cov app
