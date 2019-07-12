import click
import sys
import pandas as pd
import numpy as np  
from numpy._distributor_init import NUMPY_MKL
import re  
import nltk  
from sklearn.datasets import load_files  
nltk.download('stopwords')  
import pickle  
from nltk.corpus import stopwords  
nltk.download('wordnet')
import time
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split  
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from nltk.stem import WordNetLemmatizer

# Importing methods
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

@click.command(
    help=("Movie text sentiment analysis"))
    
@click.option(
    "--text",
    default = None,
    help="Enter your own text to test it. Please enter text in commas after --text option")
 
@click.option(
    "-t", "--train",
    nargs = 2,
    default = None,
    type=click.Path(exists=False, resolve_path=True),
    help="Enter path to 1) positive and 2) to negative reviews")
 
def main(text, train):

    click.echo("\nHello. This is movie comments sentiment analysis tool")

    if len(train) != 0:
        print("\nPositive reviews directory: {}".format(train[0]))
        print("\nNegative reviews directory: {}".format(train[1]))
        colnames =["Review"]
        pos_review = pd.read_csv("Raw data/rt-polarity_pos.csv", names=colnames, sep='|', encoding='latin-1')
        pos_review["Class"] = 1
        neg_review = pd.read_csv("Raw data/rt-polarity_neg.csv", names=colnames, sep='|', encoding='latin-1')
        neg_review["Class"] = 0
        reviews = pd.concat([pos_review, neg_review], ignore_index=True)
        X, y = reviews["Review"],reviews["Class"]
        X_raw = DocPreproc(X)
        Procesed_data = pd.DataFrame({"Review": X_raw,"Class": y})
        Procesed_data.to_csv("Preprocessed data/Preprocessed_data.csv",index=False)
    
    
        print("\nStarting RF models")
        start = time.time()
        pipe = Pipeline([("tfidfv", TfidfVectorizer(stop_words=stopwords.words('english'))),
        ("classifier", RandomForestClassifier())])
        
        parameters = {"tfidfv__max_features": (100,1000,5000),
              "tfidfv__min_df": (5,10,20),
              "tfidfv__max_df": (0.3,0.5),
              "classifier__n_estimators": (10,20,100,500,1000)
              }

        grid_rf = GridSearchCV(pipe, cv=3, n_jobs=1, param_grid=parameters)
        grid_rf.fit(X_raw, y)

        print(grid_rf.best_score_)
        print(grid_rf.best_params_)

        print("Time elapsed: {} s",round(time.time()-start,0))
    
    
        print("\nStarting KNN models")
        start = time.time()
        pipe = Pipeline([("tfidfv", TfidfVectorizer(stop_words=stopwords.words('english'))),
                 ("classifier", KNeighborsClassifier())])

        parameters = {"tfidfv__max_features": (100,1000,5000),
              "tfidfv__min_df": (5,10,20),
              "tfidfv__max_df": (0.3,0.5),
              "classifier__n_neighbors": (10,20,50,100,200),
             }

        grid_knn = GridSearchCV(pipe, cv=3, n_jobs=1, param_grid=parameters)
        grid_knn.fit(X_raw, y)

        print(grid_knn.best_score_)
        print(grid_knn.best_params_)

        print("Time elapsed: {} s".format(round(time.time()-start,0)))
        
        
        if (grid_rf.best_score_ > grid_knn.best_score_):
            print("\nRandomForest was better model")
            tfidfconverter = TfidfVectorizer(max_features=grid_rf.best_params_.get("tfidfv__max_features"),
            min_df=grid_rf.best_params_.get("tfidfv__min_df"),
            max_df=grid_rf.best_params_.get("tfidfv__max_df"),
            stop_words=stopwords.words('english'))  
            
            X = tfidfconverter.fit_transform(X_raw).toarray() 
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            classifier = RandomForestClassifier(n_estimators=grid_rf.best_params_.get("classifier__n_estimators"))          
            
        else:
            print("\nKNN was better model")
            tfidfconverter = TfidfVectorizer(max_features=grid_knn.best_params_.get("tfidfv__max_features"),
            min_df=grid_knn.best_params_.get("tfidfv__min_df"),
            max_df=grid_knn.best_params_.get("tfidfv__max_df"),
            stop_words=stopwords.words('english'))  
            
            X = tfidfconverter.fit_transform(X_raw).toarray() 
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 
        
            classifier = KNeighborsClassifier(n_neighbors=grid_knn.best_params_.get("classifier__n_neighbors")) 

        
        classifier.fit(X_train, y_train)     
        y_pred = classifier.predict(X_test)
        print("------------------------------")
        print("MODEL EVALUATION")
        print("------------------------------")
        print(confusion_matrix(y_test,y_pred))  
        print(classification_report(y_test,y_pred)) 
        print("Accuracy: ",round(accuracy_score(y_test, y_pred),2))
        print("F1 Score: ",round(f1_score(y_test, y_pred),2))
        print("Precision: ",round(precision_score(y_test, y_pred),2))
        print("Recall: ",round(recall_score(y_test, y_pred),2))
        
        
        print("------------------------------")
        print("SAVING MODEL AND VECTORIZER")
        print("------------------------------")
        with open('text_classifier', 'wb') as picklefile:  
            pickle.dump(classifier,picklefile)
        with open('converter', 'wb') as picklefile:  
            pickle.dump(tfidfconverter,picklefile)
    
    
    if text is not None:
        with open('text_classifier', 'rb') as training_model:  
            model = pickle.load(training_model)
        
        with open('converter', 'rb') as conv:  
            tfidfconverter = pickle.load(conv)
        
        model.n_jobs = 1
        
        input = ['{}'.format(text)]
 
        print("\nYou entered text: {}".format(text))
    
        X_new = tfidfconverter.transform(input).toarray()     
        y_pred = model.predict_proba(X_new)

        print("Sentiment analysis results:")
        print("--------------------------")
        print("Negative sentiment probability: {} %".format(round(y_pred[0][0]*100,2)))
        print("Positive sentiment probability: {} %".format(round(y_pred[0][1]*100,2)))            
        print("--------------------------")       
        

def DocPreproc(X):
    
    documents = []
    
    stemmer = WordNetLemmatizer()

    for sen in range(0, (len(X))):  
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(X[sen]))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        document = document.split()

        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)

        documents.append(document)
    
    return(documents)
        
if __name__=='__main__':
        
    main()
