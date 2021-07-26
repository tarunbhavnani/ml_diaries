from flask import Flask, request, redirect, url_for, flash, jsonify,render_template
from werkzeug.utils import secure_filename
import numpy as np
import pickle as p
import json
import os
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import shutil



app = Flask(__name__)

path = os.getcwd()
UPLOAD_FOLDER = os.path.join(path, 'uploads')
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#print(app.config)

try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertForSequenceClassification.from_pretrained(app.config['UPLOAD_FOLDER'])
    model.to(device)
    tok = BertTokenizer.from_pretrained(app.config['UPLOAD_FOLDER'], do_lower_case=True)
    print("model loaded")
except:
    model = None
    tok=None


@app.route("/")
def hello_world():
    return render_template('form.html')



@app.route("/upload_model", methods=['POST','PUT'])
def upload_model():
    file = request.files['file']
    filename=secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    shutil.unpack_archive(os.path.join(app.config['UPLOAD_FOLDER'], filename), app.config['UPLOAD_FOLDER'])
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global model
    model = BertForSequenceClassification.from_pretrained(app.config['UPLOAD_FOLDER'])
    model.to(device)
    global tok
    tok = BertTokenizer.from_pretrained(app.config['UPLOAD_FOLDER'], do_lower_case=True)
    print("model loaded")
    return filename

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    #print(data)
    if model:
        encoded = tok.encode_plus(data, padding="max_length", max_length=512, truncation=True, return_attention_mask=True)
        ids = torch.tensor(encoded['input_ids']).to(device)
        masks = torch.tensor(encoded['attention_mask']).to(device)
        with torch.no_grad():
            preds = model(input_ids=ids.unsqueeze(0), attention_mask=masks.unsqueeze(0))

        pred = np.argmax(preds['logits'].detach().cpu().numpy(), axis=1).item()
        #print(pred)
        return jsonify(pred)
    else:

        pred = "No Model"
        return jsonify(pred)


if __name__=='__main__':
    app.run(debug=True)
    #mdl = BertForSequenceClassification.from_pretrained(r'C:\Users\ELECTROBOT\Desktop\saved_model\full')
    #mdl.to(device)
    #tok = BertTokenizer.from_pretrained(r'C:\Users\ELECTROBOT\Desktop\saved_model\full', do_lower_case=True)

#curl -X POST -F file=@"C:\Users\ELECTROBOT\Desktop\flask_bert_ml\bert_seq_tb\full_model.zip" http://127.0.0.1:5000/upload_model
#curl --header "Content-Type: application/json; charset=utf-8"  --request POST  --data '["this is religion talk"]'  http://127.0.0.1:5000/predict