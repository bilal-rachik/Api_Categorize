import click
import pandas as pd
import pkg_resources
from flask import request
from mlflow.pyfunc import load_pyfunc
import flask

from dsflow.execution_infer.processing.abstracts.abstract import Scoringfactory

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

DSFLOW = pkg_resources.resource_filename('dsflow', '')
FILE_HTML = DSFLOW + "/templates/api_html/scoring_api.html"

app = flask.Flask(__name__)


@click.command("mode_api")
@click.option('--run_id', default='MY_RUN_ID_TEST')
@click.option('--scoring_class_name', default='assurance_vie.processing.process.AssuranceVie')
@click.option('--model', default='./ds_lib/files/artifacts/MLmodel.yml',
              help='Hdfs or Local Model path (should Zip file)')
@click.option("--port", "-p", default=5000, help="Server port. [default: 5000]")
@click.option("--host", "-h", default="0.0.0.0", help="Server host. [default: 0.0.0.0]")
def infer_click(run_id, scoring_class_name, model, port, host):
    """
    dataprocessing + scoring (Inference)
    :param run_id:
    :param scoring_class_name:
    :param model:
    :param input_file:
    :param output_file:
    :param partitions_nb:
    :return:
    """
    infer(run_id=run_id, scoring_class_name=scoring_class_name, model=model, port=port, host=host)


def infer(**kwargs):
    """
    dataprocessing + scoring (Inference)
    :param run_id:
    :param scoring_class_name:
    :param model:
    :param input_file:
    :param output_file:
    :param partitions_nb:
    :return:
    """
    scoring_class_name = kwargs['scoring_class_name']
    run_id = kwargs['run_id']
    model = kwargs['model']
    port = kwargs['port']
    host = kwargs['host']

    print('scoring_class_name = ', scoring_class_name)
    print('run_id = ', run_id)
    print('model = ', model)

    loaded_model = load_pyfunc(model)
    print('loaded_model', loaded_model)

    scoring_class = Scoringfactory.get_scoring_class(scoring_class_name)
    score_name = scoring_class.get_score_name()
    score_date = scoring_class.get_score_date()
    print('score_name', score_name)
    print('score_date', score_date)

    @app.route('/')
    def form():
        file_template = FILE_HTML
        return str(open(file_template).read())

    @app.route('/healthcheck', methods=['GET'])
    def ping():  # pylint: disable=unused-variable
        """
        Determine if the container is working and healthy.
        We declare it healthy if we can load the model successfully.
        """
        health = model is not None
        status = 200 if health else 404
        return flask.Response(response='Sucess\n', status=status, mimetype='application/json')

    @app.route('/score', methods=['POST'])
    def score():  # pylint: disable=unused-variable
        """
        Do an inference on a single batch of files. In this sample server,
        we take files as CSV or json, convert it to a pandas files frame,
        generate predictions and convert them back to CSV.
        """
        # Convert from CSV to pandas
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

        scored_df = scoring_class.score(loaded_model, df_to_score=df)
        scored_df['score_name'] = score_name
        scored_df['score_date'] = score_date
        scored_df['run_id'] = run_id
        scored_df['distribution_env'] = 'API'

        result = scored_df.to_json()
        return flask.Response(response=result, status=200, mimetype='application/json')

    @app.route('/schema', methods=['GET'])
    def schema():  # pylint: disable=unused-variable

        result = flask.jsonify(scoring_class.get_schema())
        return flask.Response(response=result, status=200, mimetype='application/json')

    app.run(port=port, host=host)


if __name__ == '__main__':