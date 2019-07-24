import pandas as pd
from OUtils import *
import os
import click
import pandas as pd
from flask import request
from mlflow.pyfunc import load_pyfunc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import click
import mlflow.sklearn


@click.command(help="Trains an PYspark model on Cdiscount dataset."
                    "The input is expected in csv format."
                    "The model and its metrics are logged with mlflow.")
#@click.option("--ngram", type=click.INT, default=1, help="ngram.")
#@click.option("--nb_hash", type=click.INT, default=1000,help="nb_hash.")
#@click.option("--maxIter", type=click.INT, default=50,help="maxIter.")
#@click.option("--regParam", type=click.FLOAT, default=0.01, help="egularization L1 .")
#@click.option("--elasticNetParam", type=click.FLOAT, default=0.0, help="Segularization L2.")
#@click.option("--word2vec", type=click.STRING, default="tf_idf",help="word2vec or tf_idf")
#@click.option("--Oversampling", type=click.BOOL, default=True, help="Oversampling")
def train():
    df = pd.read_csv("data/operation bancaire.csv")
    df.columns = ['date','libilisation','debit','credit','categories','sous categories']
    df = df.dropna(subset=['categories'])
    #typecolumns
    df = typecolumns(df)
    #train test split
    y= df['categories']
    x = df[['libilisation','credit_o_n']]
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state=42)
    Y=y_train.values
    
    # GridSearchCV to iterate over # model RandomForestClassifier
    param_grid_rf = {
        'clf__n_estimators': [700,800,900],
        'clf__max_depth' :[20,22,24,26],
        'clf__class_weight' : [None,'balanced','balanced_subsample']
    }
    #GridSearchCV to iterate over #model LogisticRegression
    param_grid_rl = {
        'clf__C': [50,70,100],
        'clf__max_iter' :[100,200,300],
        'clf__class_weight' : [None,'balanced']
    }

    with mlflow.start_run():
    
        #model RandomForestClassifier
        model_rf = MyPipeline(chose_model='rf')
        
        #model LogisticRegression
        model_rl =MyPipeline(chose_model='rl')

        #GridSearchCV and fit RandomForestclassifier
        grid_rf= GridSearchCV(model_rf, cv=10, param_grid=param_grid_rf,scoring='f1_weighted')
        grid_rf.fit(X_train,Y)
        #GridSearchCV and fit LogisticRegression
        grid_rl= GridSearchCV(model_rl, cv=10, param_grid=param_grid_rl,scoring='f1_weighted')
        grid_rl.fit(X_train,Y)
        if grid_rf.best_score_ > grid_rl.best_score_ :
            model=grid_rf
            mlflow.log_param("class_weight",grid_rf.best_params_['clf__class_weight'])
            mlflow.log_param("class_weight",grid_rf.best_params_['clf__class_weight'])
            mlflow.log_param("max_depth",grid_rf.best_params_['clf__max_depth'])
            mlflow.log_param("n_estimators",grid_rf.best_params_['clf__n_estimators'])

        else :
            model=grid_rl
            mlflow.log_param("class_weight",grid_rl.best_params_['clf__class_weight'])
            mlflow.log_param("C",grid_rl.best_params_['clf__C'])
            mlflow.log_param("max_iter",grid_rl.best_params_['clf__max_iter'])

        print("Best : %f using %s" % (model.best_score_, model.best_params_))

        # calculate f1
        y_preds=model.predict(X_test)
        f1 = f1_score(y_test, y_preds, average='weighted')
        accuracy=accuracy_score(y_test, y_preds)
        
        mlflow.log_metric("accuracy",accuracy)
        mlflow.log_metric("f1Score",f1)
        mlflow.sklearn.log_model(model,"k_models")


if __name__ == '__main__':
    train()









