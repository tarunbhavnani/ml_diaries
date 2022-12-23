from fastapi.testclient import TestClient
from .main import app
client = TestClient(app)


#async def override_dependency(q: Union[str, None]=None):
#    return {"q":q, "skip":5, "limit":10}
#app.dependency_overrides[common_parameters]= override_dependency



def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}



def test_sentiment():
    response = client.post(
        "/sentiment/",
        headers={"Content-Type": "application/json"},
        json={"text": "uhvbuhb uguyb iuhiun"},
    )
    assert response.status_code == 200
    assert response.json()["text"] == "uhvbuhb uguyb iuhiun"



