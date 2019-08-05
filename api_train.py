import pandas as pd
import os
import click
import pandas as pd
from flask import request
from mlflow.pyfunc import load_pyfunc
import flask
from io import StringIO


app = flask.Flask(__name__)

@click.command("train")
@click.option("--port", "-p", default=5000, help="Server port. [default: 5000]")
@click.option("--host", "-h", default="0.0.0.0", help="Server host. [default: 0.0.0.0]")


@app.route('/api/train', methods=['POST'])
def API():  # pylint: disable=unused-variable
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


        
    result = df.to_json()
    return flask.Response(response=result, status=200, mimetype='application/json')


if __name__=="__main__":
    app.run(debug=True)









