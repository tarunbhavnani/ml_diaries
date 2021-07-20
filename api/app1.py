from flask import Flask, request, redirect, url_for, flash, jsonify,render_template
import numpy as np
import pickle as p
import json

app= Flask(__name__)
print(app.root_path)
@app.route('/api/', methods=['POST'])
def makecalc():
	data= request.get_json()
	prediction= np.array2string(model.predict(data))

	return jsonify(prediction)


@app.route('/')
def index_page():
	return render_template('form.html')


@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    prediction = model.predict([str(text)])
    result= str(prediction.item())
    return render_template('form.html', result=result)


if __name__=='__main__':
	modelfile = './models/final_prediction_sk.pickle'
	model= np.load(open(modelfile,'rb'),allow_pickle=True)
	app.run(debug=True)
