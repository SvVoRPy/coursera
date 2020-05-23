# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 12:42:12 2017

@author: SvenV
"""

import os
path = 'C:/Users/SvenV/Desktop/Coursera Kurse/Machine Learning Spezialization III/Assignments/Week 2'
os.chdir(path)
os.listdir(path)

import pandas as pd
import numpy as np

products = pd.read_csv('amazon_baby_subset.csv')
products.iloc[0]
products['name'].iloc[1:10]

products['sentiment'].value_counts()

important_words = pd.read_json('important_words.json')[0].tolist()

# Fill in NA values for reviews with empty strings
products = products.fillna({'review':''})

import string
products['review_clean'] = products['review'].str.replace('[{}]'.format(string.punctuation), '')

for word in important_words:
    products[word] = products['review_clean'].apply(lambda s : s.split().count(word))
    
# Indicator for word perfect
products['contains_perfect'] = np.where(products['perfect'] > 0, 1, 0)
sum(products['contains_perfect'])

# Function to data matrix
def get_numpy_data(dataframe, features, label):
    dataframe['constant'] = 1
    features = ['constant'] + features
    features_frame = dataframe[features]
    feature_matrix = features_frame.as_matrix()
    label_sarray = dataframe[label]
    label_array = label_sarray.as_matrix()
    return(feature_matrix, label_array)

feature_matrix, sentiment = get_numpy_data(products, important_words, 'sentiment')
feature_matrix.shape


'''
produces probablistic estimate for P(y_i = +1 | x_i, w).
estimate ranges between 0 and 1.
'''
def predict_probability(feature_matrix, coefficients):
    # Take dot product of feature_matrix and coefficients      
    score = np.dot(feature_matrix,coefficients)
    # Compute P(y_i = +1 | x_i, w) using the link function
    predictions = np.array(1/(1+np.exp(-score)))
    # return predictions
    return predictions

def feature_derivative(errors, feature):     
    # Compute the dot product of errors and feature
    derivative = np.dot(errors,feature).sum()
        # Return the derivative
    return derivative

def compute_log_likelihood(feature_matrix, sentiment, coefficients):
    indicator = (sentiment==+1)
    scores = np.dot(feature_matrix, coefficients)
    lp = np.sum((indicator-1)*scores - np.log(1 + np.exp(-scores)))
    return lp
    

from math import sqrt
def logistic_regression(feature_matrix, sentiment, initial_coefficients, step_size, max_iter):
    coefficients = np.array(initial_coefficients) # make sure it's a numpy array
    for itr in xrange(max_iter):
        predictions = predict_probability(feature_matrix, coefficients)
        indicator = np.where(products['sentiment'] == 1, 1, 0)
        errors = indicator - predictions

        for j in xrange(len(coefficients)): 
            #print "Iteration " + str(itr) + " updating weight " + str(j)
            derivative = feature_derivative(errors, feature_matrix[:,j])
            coefficients[j] = coefficients[j] + np.dot(step_size,derivative)

        # Checking whether log likelihood is increasing
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
        or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            lp = compute_log_likelihood(feature_matrix, sentiment, coefficients)
            print 'iteration %*d: log likelihood of observed labels = %.8f' % \
                (int(np.ceil(np.log10(max_iter))), itr, lp)
    return coefficients   
    
initial_coefficients = np.repeat(float(0),feature_matrix.shape[1])
step_size = 1e-7 
max_iter = 301

coefficients = logistic_regression(feature_matrix, sentiment, initial_coefficients, step_size, max_iter)

scores = np.dot(feature_matrix,coefficients)

pd.Series(np.where(scores > 0, 1, -1)).value_counts()

pd.Series(sentiment == pd.Series(np.where(scores > 0, 1, -1))).value_counts()[1]/float(len(sentiment))

coefficients = list(coefficients[1:])
word_coefficient_tuples = [(word, coefficient) for word, coefficient in zip(important_words, coefficients)]
word_coefficient_tuples = sorted(word_coefficient_tuples, key=lambda x:x[1], reverse=True)

word_coefficient_tuples[0:9]

word_coefficient_tuples[-20:]




