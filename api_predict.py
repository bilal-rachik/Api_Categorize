
import pandas as pd
import mlflow.pyfunc
from flask import Flask, request, json
import numpy as np
from OUtils import *
import flask
import mlflow.sklearn

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/api/categorize', methods=['POST'])
def pred():
    df = read_data(request.json)
    model_path_dir = "top_model"
    model = mlflow.sklearn.load_model(model_path_dir)
    prd = model.predict(df)
    df["predict"]=prd
    df=df.drop(['credit_o_n'],axis='columns')
    df = df.to_json(orient='records')
    return flask.Response(response=df, status=200, mimetype='application/json')
if __name__ == "__main__":
    app.run(debug=True,port=6060)