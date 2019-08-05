from sklearn.base import BaseEstimator, TransformerMixin
import nltk
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline 
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import mlflow.sklearn
import os
import shutil
nltk.download("all")
def typecolumns(df):
    df.debit[df.debit==""]=np.NaN
    df.credit[df.credit==""]=np.NaN
    df['debit'] = df.debit.astype(float)
    df['credit'] = df.credit.astype(float)
    df['credit_o_n']=df.credit.isnull()
    df['libilisation']=df.libilisation.astype(str)
    return df

#stop word
def stopwords():
    s = set(nltk.corpus.stopwords.words('french'))
    lucene_stopwords = open("dataOutil/lucene_stopwords.txt","r").read().split(",") #En local
    ## Union des deux fichiers de stopwords 
    stopwords = list(s.union(set(lucene_stopwords)))
    return stopwords


def read_data(data):
    df = pd.DataFrame.from_dict(data, orient='columns')
    df.columns = ['credit', 'date', 'debit', 'libilisation']
    # typecolumns
    df = typecolumns(df)
    return df
 
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None,**fit_params):
        return self
    def transform(self, X):
        return X[self.attribute_names].values 
    
class MyLabelBinar(LabelBinarizer):
    def fit(self, X, y=None):
        """this would allow us to fit the model based on the X input."""
        super(MyLabelBinar, self).fit(X)
    def transform(self, X, y=None):
        return super(MyLabelBinar, self).transform(X)

    def fit_transform(self, X, y=None):
        return super(MyLabelBinar, self).fit(X).transform(X)
    
def get_merged_csv(flist, **kwargs):
    return pd.concat([pd.read_csv(f, **kwargs) for f in flist], ignore_index=True)

def MyPipeline(chose_model='rf',C=50,max_iter = 200,class_weight = None,n_estimators = 500,max_depth = 20):
    num_libilisation = 'libilisation'
    num_credit_oui_non = 'credit_o_n'
    #tfidf
    tfv = TfidfVectorizer(min_df=1,max_features=None,stop_words=stopwords(), strip_accents='unicode',
        analyzer="word", token_pattern=r'\b[a-zA-Z]{3,30}\b',ngram_range=(1,1),norm = False, sublinear_tf=1)
        
    features = FeatureUnion([
                    ('credit_oui_non', Pipeline([('selector',DataFrameSelector(num_credit_oui_non)),('encoder',MyLabelBinar())])),
                    ('words', Pipeline([('selector', DataFrameSelector(num_libilisation)),('wtfidf',tfv)]))]) 
    #model
    if chose_model =='rl':   
        cla = LogisticRegression(C=C,n_jobs=-1,max_iter=max_iter,class_weight=class_weight,solver='newton-cg',multi_class='multinomial')
        model = Pipeline([  
            ('features',features),('clf',cla)])
    else :
        f = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,class_weight=class_weight,n_jobs=-1)
        model = Pipeline([  
            ('features',features),('clf',f)])
    return model

def start_train(df):
    df.columns = ['date', 'libilisation', 'debit', 'credit', 'categories', 'sous categories']
    df = df.dropna(subset=['categories'])
    # typecolumns
    df = typecolumns(df)
    # train test split
    y = df['categories']
    x = df[['libilisation', 'credit_o_n']]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    Y = y_train.values

    # GridSearchCV to iterate over # model RandomForestClassifier
    param_grid_rf = {
        'clf__n_estimators': [600,700,800],
        'clf__max_depth': [20,26,28],
        'clf__class_weight': [None, 'balanced', 'balanced_subsample']
    }
    # GridSearchCV to iterate over #model LogisticRegression
    param_grid_rl = {
        'clf__C': [50,80,100],
        'clf__max_iter': [200,300,400],
        'clf__class_weight': [None, 'balanced']
    }
    with mlflow.start_run():
        # model RandomForestClassifier
        model_rf = MyPipeline(chose_model='rf')
        # model LogisticRegression
        model_rl = MyPipeline(chose_model='rl')
        # GridSearchCV and fit RandomForestclassifier
        grid_rf = GridSearchCV(model_rf, cv=10, param_grid=param_grid_rf, scoring='f1_weighted')
        grid_rf.fit(X_train, Y)
        # GridSearchCV and fit LogisticRegression
        grid_rl = GridSearchCV(model_rl, cv=10, param_grid=param_grid_rl, scoring='f1_weighted')
        grid_rl.fit(X_train, Y)
        mod=""
        if grid_rf.best_score_ > grid_rl.best_score_:
            mod="RandomForestClassifier"
            grid=grid_rf
            model = MyPipeline(chose_model='rf',class_weight=grid_rf.best_params_['clf__class_weight'],max_depth=grid_rf.best_params_['clf__max_depth'],n_estimators=grid_rf.best_params_['clf__n_estimators'])
            model.fit(X_train,Y)
            mlflow.log_param("class_weight", grid_rf.best_params_['clf__class_weight'])
            mlflow.log_param("max_depth", grid_rf.best_params_['clf__max_depth'])
            mlflow.log_param("n_estimators", grid_rf.best_params_['clf__n_estimators'])
        else:
            mod = "LogisticRegression"
            grid= grid_rf
            model = MyPipeline(chose_model='rl',class_weight=grid_rl.best_params_['clf__class_weight'],C=grid_rl.best_params_['clf__C'],max_iter=grid_rl.best_params_['clf__max_iter'])
            model.fit(X_train,Y)
            mlflow.log_param("class_weight", grid_rl.best_params_['clf__class_weight'])
            mlflow.log_param("C", grid_rl.best_params_['clf__C'])
            mlflow.log_param("max_iter", grid_rl.best_params_['clf__max_iter'])
        # calculate f1
        y_preds = model.predict(X_test)
        f1 = f1_score(y_test, y_preds, average='weighted')
        accuracy = accuracy_score(y_test, y_preds)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1Score", f1)
        mlflow.sklearn.log_model(model, "k_models")
        if os.path.exists('top_model'):
            shutil.rmtree('top_model')
        mlflow.sklearn.save_model(model, 'top_model', serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE)
        return "model : %s Best : %f using %s" % (mod,f1,grid.best_params_)