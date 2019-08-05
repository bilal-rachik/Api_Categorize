import pandas as pd
import os
from OUtils import *
import click
import pandas as pd
from flask import request
from mlflow.pyfunc import load_pyfunc
import flask
from io import StringIO
app = flask.Flask(__name__)
import glob

@click.command("train")
@click.option("--port", "-p", default=5000, help="Server port. [default: 5000]")
@click.option("--host", "-h", default="0.0.0.0", help="Server host. [default: 0.0.0.0]")
def start(port,host):
    
    @app.route('/healthcheck', methods=['GET'])
    def ping():  # pylint: disable=unused-variable
        """
        Determine if the container is working and healthy.
        We declare it healthy if we can load the model successfully.
        """
        health = os.path.exists('top_model')
        status = 200 if health else 404
        return flask.Response(response='Sucess\n', status=status, mimetype='application/json')

    @app.route('/api/addcsv', methods=['POST'])
    def addcsv():
        if flask.request.content_type == 'text/csv':
            data = request.data.decode('utf-8')
            s = StringIO(data)
            df = pd.read_csv(s)

        if flask.request.content_type == 'application/json':
            # json = flask.request.get_json()
            flask.request.get_json()
            json = flask.request.get_json()
            df = pd.DataFrame((json))

        if str(flask.request.content_type).__contains__("multipart/form-files"):
            print("---------------------- score csv file ------------------------")
            file = request.files['file']
            file_contents = file.stream
            df = pd.read_csv(file_contents)
        if str(flask.request.content_type).__contains__("application/x-www-form-urlencoded"):
            data = request.form['json_data']
            df = pd.DataFrame(eval(data), index=[0])
        flist = glob.glob('data/*.csv')
        df.to_csv('data/operation banciare_{}.csv'.format(len(flist)),index=False, header=True)
        return flask.Response(response="sucess", status=200, mimetype='application/json')

    @app.route('/api/train')
    def train():
        flist = glob.glob('data/*.{}'.format('csv'))
        df = get_merged_csv(flist)
        result = start_train(df)
        return flask.Response(response=result, status=200, mimetype='application/json')
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
    app.run(port=port)

if __name__=="__main__":
    start()





