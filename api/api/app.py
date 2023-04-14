from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
from flask_restful import Resource, Api
import pickle as p
import json

app = Flask(__name__)
api = Api(app)


# print(app.root_path)
# @app.route('/api/', methods=['POST'])
class makecalc(Resource):
	def post(self):
		data = request.get_json()
		prediction = str(model.predict(data).item())
		return jsonify(prediction)


api.add_resource(makecalc, '/api/')
# prediction= np.array2string(model.predict(data))
if __name__ == '__main__':
    modelfile = './models/final_prediction_sk.pickle'
    model = np.load(open(modelfile, 'rb'), allow_pickle=True)
    app.run(debug=True)
