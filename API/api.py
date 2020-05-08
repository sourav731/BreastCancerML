from flask import Flask,jsonify,request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

filename = '../breastcancer.pkl'
with open(filename,'rb') as file:
    model = pickle.load(file)

filename = '../breastcancercolumns.pkl'
with open(filename,'rb') as file:
    columns = pickle.load(file)

@app.route("/")
def hello():
    return "Welcome to machine learning model API!"

@app.route("/predict",methods=['POST'])
def predict():
    if model:
        try:
            json_ = request.json

            try:
                query = pd.get_dummies(pd.DataFrame(json_))
            except:
                return jsonify({'error' : 'something went wrong'})

            query = query.reindex(columns=columns, fill_value=0)
            prediction = model.predict(query)

            return jsonify({'prediction': str(prediction)})

        except:
            return jsonify({'error' : 'something went wrong'})
    else:
        return jsonify({'error' : 'Model not loaded'})


if __name__ == '__main__':
    app.run(debug=True)

