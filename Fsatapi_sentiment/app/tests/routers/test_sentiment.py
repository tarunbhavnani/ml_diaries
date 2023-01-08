import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
from app.main import app
from app.functions import get_sentiment
client = TestClient(app)


def test_get_sentiment_scores():
    response= client.post("/sentiment", json={"text":"Good"})
    assert response.status_code == 200
    assert response.json()["text"]=="Good"



