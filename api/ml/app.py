from flask import Flask, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import pickle as p
import json
import os

path = os.getcwd()
print(path)
app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(path, 'uploads')
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
try:
    modelfile = os.path.join(path, 'uploads/final_prediction.pickle')
    model = p.load(open(modelfile, 'rb'))
    print("model loaded")
except:
    model = None


@app.route("/")
def hello_world():
    return "Hello World"


@app.route('/api/', methods=['POST'])
def makecalc():
    data = request.get_json()
    if model:
        prediction = np.array2string(model.predict(data))

        return jsonify(prediction)
    else:

        prediction = "No Model"
        return jsonify(prediction)


@app.route("/upload_model", methods=['POST','PUT'])
def upload_model():
    file = request.files['file']
    filename=secure_filename(file.filename)
    #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    #modelfile = os.path.join(path, 'uploads/final_prediction.pickle')
    global model
    #model = p.load(open(modelfile, 'rb'))
    model = p.load(file.stream)
    print("model loaded")

    return filename


if __name__ == '__main__':
    # modelfile='models/final_prediction.pickle'
    # model= p.load(open(modelfile,'rb'))
    app.run(debug=True)
#curl --header "Content-Type: application/json"  --request POST  --data "[[4.34, 1.68, 0.7, 5.0, 8.0, 2.8, 1.31, 0.53, 2.7, 13.0, 0.57, 1.96, 60.0]]"  http://127.0.0.1:5000/api/
#curl -X POST -F file=@"C:\Users\ELECTROBOT\Desktop\upload_api\ml\models\final_prediction.pickle" http://127.0.0.1:5000/upload_model
#curl --header "Content-Type: application/json"  --request POST  --data "[[4.34, 1.68, 0.7, 5.0, 8.0, 2.8, 1.31, 0.53, 2.7, 13.0, 0.57, 1.96, 60.0]]"  http://127.0.0.1:5000/api/