import sys

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
import re
import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import fbeta_score,classification_report,make_scorer
from sklearn.utils.multiclass import type_of_target
import pickle



class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """Custom pipeline class for extracting features from data."""
    
    def starting_verb(self, text):
        """Returns true if sentense starts with verb"""
        
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if len(pos_tags) != 0:
                first_word, first_tag = pos_tags[0]
                # return true of sentence start with verb
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def load_data(database_filepath):
    """
    Load data from sqlite database.
    
    Returns:
    df (pandas dataframe): loaded dataset
    X: messages column
    Y: classification categories
    category_names: containes column names of classification categories
    """
    
    database = 'sqlite:///'+database_filepath
    engine = create_engine(database)
    df = pd.read_sql_table('disaster_response', database)  
    
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = list(df.columns[4:])
    return df, X, Y, category_names 


def tokenize(text):
    """remove punctuation from text, tokenize and extract words. """
   
    # remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    #tokenize text and extrac words
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        #lower case and remove extra spaces
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def get_pipeline():
    """
    Creates machine learning pipeline.
    
    The pipeline takes in the `message` column as input and output classification results on the other 36 categories in the dataset. 
    """
    
    pipeline = Pipeline([
                        ('features',FeatureUnion([
                                                 ('text-pipeline',Pipeline([
                                                                            ('vect', CountVectorizer(tokenizer= tokenize)),
                                                                            ('tfidf', TfidfTransformer())
                                                                           ])),
                                                 ('starting-verb',StartingVerbExtractor())
                                                 ])),
                        ('clf', MultiOutputClassifier(RandomForestClassifier()))
                        ])
    return pipeline

def build_model():
    """Use GridSearch to find better parameters to tune the model."""
    
    pipeline = get_pipeline()

    #parameters for grid search (based on pipeline parameters)
    parameters = {  
                        'clf__estimator__min_samples_split': [2, 4],
                        #'clf__n_estimators': [50, 100, 200],
                        #'features__text-pipeline__tfidf__use_idf' : [True, False],
                        #'clf__estimator__max_depth': [100, 150, 200],
                        #'clf__estimator__max_features': ['auto', 'sqrt','log2']
                }

    cv = GridSearchCV(estimator=pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Outputs the overall accuracy and a classification report for all 36 categories.
    
    The classification report contains: 'precision', 'recall', 'f1-score' , 'support'.
    """
    Y_pred = model.predict(X_test)

    #get mean accuracy
    accuracy = (Y_pred == Y_test).mean().mean()
    print('Overall accuracy {}% \n'.format(accuracy*100))

    #generate report for each classification category
    Y_pred_df = pd.DataFrame(Y_pred, columns = category_names)
    for column in category_names:
        print(classification_report(Y_test[column],Y_pred_df[column]))


def save_model(model, model_filepath):
    """save model as pickle file"""
    pickle.dump(model, open(model_filepath, 'wb'))


