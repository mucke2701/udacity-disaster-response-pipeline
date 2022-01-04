import sys

# import libraries
import nltk
nltk.download('punkt')
nltk.download('wordnet')
#nltk.download()
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import re
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import multioutput
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
nltk.download('stopwords')
from sklearn.metrics import fbeta_score, make_scorer
import pickle

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql ('SELECT * FROM '+  os.path.basename(database_filepath).replace(".db","") + "_table", engine)
    #display (df.head (n=10))
    X = df ['message']
    y = df.iloc[:,4:]
    category_names = y.columns
    return X,y, category_names


def tokenize(text):
    # import statements
    stop_words = stopwords.words("english")
    
    #tokenize
    words = word_tokenize (text)
    
    #stemming
    stemmed = [PorterStemmer().stem(w) for w in words]
    
    #lemmatizing
    words_lemmed = [WordNetLemmatizer().lemmatize(w) for w in stemmed if w not in stop_words]
   
    return words_lemmed


def build_model():
    #setting pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', multioutput.MultiOutputClassifier (GradientBoostingClassifier()))
        ])
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    avg_f1score = 0
    for i in range(len(Y_pred)):
        avg_f1score = avg_f1score + f1_score(Y_test, Y_pred, zero_division=1)
    avg_f1score = avg_f1score/len(Y_pred)
    print('Average F1-Score: ' + avg_f1score)



def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
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