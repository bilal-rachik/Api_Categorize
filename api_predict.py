
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


df=[
  {
    "date_transaction": "2018-12-28",
    "lib1": "LE FIVE VILLETT AUBERVILLIER",
    "debit": 70,
    "credit": ""
  },
  {
    "date_transaction": "2019-01-02",
    "lib1": "VIR MAAF ASSURANCES SA 0309201805911281 VIRT 10 MAAF AS REMB192367366",
    "debit": "",
    "credit": 2205.5
  },
  {
    "date_transaction": "2019-01-07",
    "lib1": "AMAZON EU SARL PAYLI2090401",
    "debit": 759,
    "credit": ""
  },
  {
    "date_transaction": "2019-01-07",
    "lib1": "UNT BORDEAUX 120-126 QUAI DE /INV/20190104 C/C05P19004120611",
    "debit": "",
    "credit": 30936
  },
  {
    "date_transaction": "2019-01-07",
    "lib1": "AMAZON PAYMENTS PAYLI2441535",
    "debit": 549.23,
    "credit": ""
  },
  {
    "date_transaction": "2019-01-17",
    "lib1": "VIR GPE14 487 556",
    "debit": -431.48,
    "credit": ""
  },
  {
    "date_transaction": "2019-01-18",
    "lib1": "CARTE X5952 16/01 BASILIC VILLETT",
    "debit": -74,
    "credit": ""
  }]






@app.route('/api/categorize', methods=['POST'])
def pred():
    df = pd.DataFrame.from_dict(request.json,orient='columns')
    df.columns = ['credit','date','debit','libilisation']
    # typecolumns
    df = typecolumns(df)
    model_path_dir = "mlruns/0/cc49a312b5b94703b3447e4a2ab7e8f9/artifacts/k_models"
    model = mlflow.sklearn.load_model(model_path_dir)
    prd = model.predict(df)
    df["predict"]=prd
    df=df.drop(['credit_o_n'],axis='columns')
    df = df.to_json(orient='records')
    return flask.Response(response=df, status=200, mimetype='application/json')


if __name__ == "__main__":
    app.run(debug=True,port=6060)