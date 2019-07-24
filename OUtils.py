from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import nltk
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline 
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import RandomForestClassifier
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
    lucene_stopwords = open("data/lucene_stopwords.txt","r").read().split(",") #En local
    ## Union des deux fichiers de stopwords 
    stopwords = list(s.union(set(lucene_stopwords)))
    return stopwords
 
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
    
    
def MyPipeline(chose_model='rf',balanced = False,max_depth=15):
    
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
        cla = LogisticRegression(C=50,n_jobs=-1,max_iter=200,solver='newton-cg',multi_class='multinomial')
        model = Pipeline([  
            ('features',features),('clf',cla)])
    else :
        f = RandomForestClassifier(n_estimators=200,n_jobs=-1)
        model = Pipeline([  
            ('features',features),('clf',f)])
    return model 


 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    