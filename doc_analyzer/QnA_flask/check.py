import requests

resp = requests.post("http://localhost:3995/predict",
                     files={"file": "who is roger federer"})
resp.json()