from flask import Flask, request, redirect, url_for, flash, jsonify
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

if __name__=='__main__':
	modelfile='./models/final_prediction.pickle'
	model= np.load(open(modelfile,'rb'),allow_pickle=True)
	app.run(debug=True)