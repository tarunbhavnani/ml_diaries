import requests
import json

url = 'http://127.0.0.1:5000/api/'

# data = [[14.34, 1.68, 2.7, 25.0, 98.0, 2.8, 1.31, 0.53, 2.7, 13.0, 0.57, 1.96, 660.0]]
data = [[4.34, 1.68, .7, 5.0, 8.0, 2.8, 1.31, 0.53, 2.7, 13.0, 0.57, 1.96, 60.0]]
j_data = json.dumps(data)

headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}

r = requests.post(url, data=j_data, headers=headers)

print(r, r.text, "tarun")

# put data in file

import json

with open(r"C:\Users\ELECTROBOT\Desktop\body.json", 'w') as fp:
    json.dump(data, fp)

curl - H
"Content-Type: application/json" - -data @ body.json
http: // 127.0
.0
.1: 5000 / api /

# or put data in command

curl - -header
"Content-Type: application/json" - -request
POST - -data
"[[4.34, 1.68, 0.7, 5.0, 8.0, 2.8, 1.31, 0.53, 2.7, 13.0, 0.57, 1.96, 60.0]]"
http: // 127.0
.0
.1: 5000 / api /
