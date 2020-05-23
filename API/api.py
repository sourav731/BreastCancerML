from flask import Flask,jsonify,request,make_response
import pickle
import numpy as np
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)
app = Flask(__name__)

filename = '../breastcancer.pkl'
with open(filename,'rb') as file:
    model = pickle.load(file)

filename = '../breastcancercolumns.pkl'
with open(filename,'rb') as file:
    columns = pickle.load(file)

def build_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response
def build_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

@app.route("/")
def hello():
    return "Welcome to machine learning model API!"

@app.route("/predict",methods=['OPTIONS','POST'])
def predict():
	if request.method == 'OPTIONS': 
        	return build_preflight_response()

	elif request.method == 'POST':
    		if model:
        		try:
            			json_ = request.json

            			try:
                			query = pd.get_dummies(pd.DataFrame(json_))
            			except:
                			return jsonify({'error' : 'something went wrong'})

            			query = query.reindex(columns=columns, fill_value=0)
            			prediction = model.predict(query)

            			return build_actual_response(jsonify({'prediction': str(prediction)}))

        		except:
            			return jsonify({'error' : 'something went wrong'})


if __name__ == '__main__':
    app.run(debug=True)

