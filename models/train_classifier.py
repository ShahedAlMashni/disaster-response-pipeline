from helper import *
import numpy as np
import pandas as pd

# class StartingVerbExtractor(BaseEstimator, TransformerMixin):

#     def starting_verb(self, text):
#         sentence_list = nltk.sent_tokenize(text)
#         for sentence in sentence_list:
#             pos_tags = nltk.pos_tag(tokenize(sentence))
#             if len(pos_tags) != 0:
#                 first_word, first_tag = pos_tags[0]
#                 if first_tag in ['VB', 'VBP'] or first_word == 'RT':
#                     return True
#         return False

#     def fit(self, x, y=None):
#         return self

#     def transform(self, X):
#         X_tagged = pd.Series(X).apply(self.starting_verb)
#         return pd.DataFrame(X_tagged)

# def load_data(database_filepath):
#     database = 'sqlite:///'+database_filepath
#     engine = create_engine(database)
#     df = pd.read_sql_table('disaster_response', database)  
    
#     X = df['message']
#     Y = df.iloc[:, 4:]
#     category_names = list(df.columns[4:])
#     return X,Y,category_names 


# def tokenize(text):
#     text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
#     tokens = word_tokenize(text)
#     lemmatizer = WordNetLemmatizer()

#     clean_tokens = []
#     for tok in tokens:
#         clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#         clean_tokens.append(clean_tok)

#     return clean_tokens

# def get_pipeline():
    
#     pipeline = Pipeline([
#                         ('features',FeatureUnion([
#                                                  ('text-pipeline',Pipeline([
#                                                                             ('vect', CountVectorizer(tokenizer= tokenize)),
#                                                                             ('tfidf', TfidfTransformer())
#                                                                            ])),
#                                                  ('starting-verb',StartingVerbExtractor())
#                                                  ])),
#                         ('clf', MultiOutputClassifier(RandomForestClassifier()))
#                         ])
#     return pipeline

# def build_model():
#     pipeline = get_pipeline()
#     parameters = {  
#                         'clf__estimator__min_samples_split': [2, 4],
#                         'clf__n_estimators': [50, 100, 200],
#                         'features__text-pipeline__tfidf__use_idf' : [True, False],
#                         'clf__estimator__max_depth': [100, 150, 200],
#                         'clf__estimator__max_features': ['auto', 'sqrt','log2']
#                 }

#     cv = GridSearchCV(estimator=pipeline, param_grid=parameters)
    
#     return cv

# def evaluate_model(model, X_test, Y_test, category_names):
#     Y_pred = model.predict(X_test)
#     accuracy = (Y_pred == Y_test).mean().mean()
#     print('Overall accuracy {}% \n'.format(accuracy*100))

#     Y_pred_df = pd.DataFrame(Y_pred, columns = category_names)
#     for column in category_names:
#         print(classification_report(Y_test[column],Y_pred_df[column]))


# def save_model(model, model_filepath):
#     pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:

        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))

        df,X, Y, category_names = load_data(database_filepath)
        #split data into train and test datasets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()