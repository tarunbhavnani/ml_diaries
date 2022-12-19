from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello World"}


def test_reset():
    response = client.get("/reset/")
    assert response.status_code == 200
    assert response.json() == {"file_name": "Good"}




def test_predict():
    response = client.post(
        "/predict/",
        headers={"Content-Type": "application/json"},
        json={"text": "uhvbuhb uguyb iuhiun"},
    )
    assert response.status_code == 200
    assert response.json() == { "class": "No files"}


