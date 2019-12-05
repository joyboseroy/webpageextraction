# -*- coding: utf-8 -*-
"""
Build model for extraction using ML
Created on Thu Sep 20 11:04:38 2018
Build machine learning model out of the extracted features

@author: jobose
"""

# Author: Joy Bose 

#%matplotlib inline

import numpy as np 
import matplotlib.pyplot as plt
import pickle
import matplotlib.pylab as plt
import csv
import pandas as pd
import requests

from sklearn.datasets import load_digits
from sklearn.externals import joblib

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics, neighbors
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from scipy.misc.pilutil import imresize

#import extraction_ML_dataset_write
from bs4 import BeautifulSoup, SoupStrainer
from collections import Counter


def read_train_dataset(dataset_path, model_pickle_filename):
    """Read the dataset to pandas dataframe 
    AFTER the manual labelling is completed
    
    Fit linear SVM model to the dataset
    Test and get precision recall

    Parameters
    ----------
    dataset_path : string
        Path to the dataset file.
    
    model_pickle_filename : string
        Name of the pickle file (with path if needed) stored on disk.

    Returns
    -------
    None
    """

    """
    df = pd.read_csv(dataset_path, 
                     header=None, 
                     names=['url', 
                            'word_count', 
                            'input_count', 
                            'script_count', 
                            'link_input_script_count', 
                            'text_density', 
                            'link_density',
                            'label'])
    """
    df = pd.read_csv(dataset_path, header = None)
    #print(df.dtypes)
    #print(df.describe(include='all'))
    #print(df.head())
    print('Dataframe shape is = ', df.shape)
    
    X = df[df.columns[1:7]]
    y= df[df.columns[7]] #predicted value
    print(X)
    
    #Split data into train and test set in 80-20 ratio
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    print('Result with linear SVM')
    linearsvm = svm.LinearSVC()
    linearsvm.fit(x_train, y_train)

    #save trained SVM model to disk as pickle file
    save_model_pickle(linearsvm, model_pickle_filename)

    y_pred = linearsvm.predict(x_test)
    print(classification_report(y_test, y_pred))

def predict_readermode_file(input_HTML_filename, model_pickle_filename, output_HTML_filename):
    """Using the stored model, 
    Predict the reading view HTML for a new HTML file
    Copy the original HTML if the predicted value for the <p> block is 1

    Parameters
    ----------
    input_HTML_filename : string
        Name of the HTML file (wth path if not in same dir) to read and parse.
    
    model_pickle_filename : string
        Name of the pickle file (with path if needed) stored on disk.
    
    output_HTML_filename : string
        Name of the reading view output HTML file.
    
    Returns
    -------
    None
    """    
    model = load_model_pickle(model_pickle_filename)
    html_file =  open(input_HTML_filename, 'rb') 
    readingview_file = open(output_HTML_filename, 'a', encoding = 'utf-8')
    
    readingview_file.write('<html><body>')
    
    #extraction_ML_dataset_write.parse_file()
    #Use BeautifulSoup to parse the file
    soup = BeautifulSoup(html_file, 'lxml') #, parse_only = strainer) #'html.parser')
    
    #line = extraction_ML_dataset_write.extract_features(soup)
    mylist = soup.find_all(re.compile(r'h1|h2|h3|h4|p|img'))
    for items in mylist:
        link_count = 0
        for link in items.find_all('a'):
            link_count = link_count + 1
        
        word_count = len(items.text.split())
        input_count = len(items.find_all('input'))
        script_count = len(items.find_all('script'))
        
        #get previous link and text density?
        
        link_input_script_count = link_count + input_count + script_count
        links_plus_words = link_input_script_count + word_count
        if links_plus_words != 0:
            link_density = link_input_script_count/links_plus_words
            text_density = word_count/links_plus_words
        else:
            link_density = 0
            text_density = 0
        
        X = [word_count,\
             input_count,\
             script_count,\
             link_input_script_count,\
             round(text_density, 2),\
             round(link_density, 2)]
        
        #print('X vector is = ', X)
        content_or_noise = model.predict([X])
        #print('Predicted value is=', content_or_noise[0])
        if content_or_noise[0] == 1:
            #write p block as such to the output file
            print(str(items) + '\n')
            readingview_file.write(str(items))

    readingview_file.write('</body></html>')
        
    readingview_file.close()
    html_file.close()
    
def save_model_joblib(linearsvm, save_model_filename):
    """Save the classifier model as a PKL file using joblib
    
    Parameters
    ----------
    linearsvm : linear SVM object
        Initialized SVM object (SVC with parameter linear).
    
    save_model_filename : string
        Name of the pickle file of SVM model to save on disk.
    
    Returns
    -------
    None    
    """
    joblib.dump(linearsvm, save_model_filename, compress = 9)

def load_model_joblib(model_file):
    """Load previously saved PKL model file from disk using joblib

    Parameters
    ----------
    model_file : model object
        SVM model Object.
    
    Returns
    -------
    None    """
    model = joblib.load(model_file)
    return model
    
def save_model_pickle(model, save_model_filename):
    """Save the classifier model pickle file using pickle.dump

    Parameters
    ----------
    model : linear SVM object
        Initialized SVM object (SVC with parameter linear).
    
    save_model_filename : string
        Name of the pickle file of SVM model to save on disk.
    
    Returns
    -------
    None    
    """
    with open(save_model_filename, 'wb') as f: 
        pickle.dump(model, f)

def load_model_pickle(model_filename):
    """Load previously saved Pickle file using pickle.load

    Parameters
    ----------
    model_filename : string
        Name of the pickle file of SVM model previously saved on disk.
    
    Returns
    -------
    None    
    """
    with open(model_filename, 'rb') as f: 
        model = pickle.load(f)
    return model
    
def predict_data(svm_model, x_test, y_test):
    """Using the loaded model and the input data, predict the test data 
    Print classification report for the results

    Parameters
    ----------
    svm_model : linear SVM object
        Initialized SVM object (SVC with parameter linear).
        
    x_test : {array-like, sparse matrix}, shape (n_samples, n_features)
        Test data.
        
    y_test : ndarray, shape (n_samples,)
        Array of labels.

    Returns
    -------
    None    

    """
    y_pred = svm_model.predict(x_test)
    print(classification_report(y_test, y_pred))    
    
def main():
    dataset_path = 'dataset_labelled.csv'
    pickle_filename = 'readingview_classifier.pkl'

    #myhtml_filename = 'myfile_BBC.html'
    #outputhtml_filename = 'myfile_readermode_BBC.html'
    #read_train_dataset(dataset_path, pickle_filename)
    #predict_readermode_file(myhtml_filename, pickle_filename, outputhtml_filename)

    #myhtml_filename = 'myfile_BBC1.html'
    #outputhtml_filename = 'myfile_readermode_BBC1.html'
    #read_train_dataset(dataset_path, pickle_filename)
    #predict_readermode_file(myhtml_filename, pickle_filename, outputhtml_filename)

    myhtml_filename = 'myfile_Wikipedia.html'
    outputhtml_filename = 'myfile_readermode_Wikipedia.html'
    read_train_dataset(dataset_path, pickle_filename)
    predict_readermode_file(myhtml_filename, pickle_filename, outputhtml_filename)

    #myhtml_filename = 'myfile_TOI.html'
    #outputhtml_filename = 'myfile_readermode_TOI.html'
    #read_train_dataset(dataset_path, pickle_filename)
    #predict_readermode_file(myhtml_filename, pickle_filename, outputhtml_filename)

    #myhtml_filename = 'myfile_CNN.html'
    #outputhtml_filename = 'myfile_readermode_CNN.html'
    #read_train_dataset(dataset_path, pickle_filename)
    #predict_readermode_file(myhtml_filename, pickle_filename, outputhtml_filename)

if __name__ == '__main__': 
    main()    