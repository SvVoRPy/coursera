# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 07:46:06 2018

@author: SvenV
"""

import os
path = 'C:/Users/SvenV/Desktop/Coursera Kurse/Machine Learning Spezialization III/Assignments/Week 7'
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
    print(word)
    products[word] = products['review_clean'].apply(lambda s : s.split().count(word))
    
# Train Validation Split
train_data = products.iloc[pd.read_json('module-10-assignment-train-idx.json')[0],:]
validation_data = products.iloc[pd.read_json('module-10-assignment-validation-idx.json')[0],:]

def get_numpy_data(dataframe, features, label):
    dataframe['constant'] = 1
    features = ['constant'] + features
    features_frame = dataframe[features]
    feature_matrix = features_frame.as_matrix()
    label_sarray = dataframe[label]
    label_array = label_sarray.as_matrix()
    return(feature_matrix, label_array)
    
feature_matrix_train, sentiment_train = get_numpy_data(train_data, important_words, 'sentiment')
feature_matrix_valid, sentiment_valid = get_numpy_data(validation_data, important_words, 'sentiment')

def predict_probability(feature_matrix, coefficients):
    # Take dot product of feature_matrix and coefficients      
    score = np.dot(feature_matrix,coefficients)
    # Compute P(y_i = +1 | x_i, w) using the link function
    predictions = np.array(1/(1+np.exp(-score)))
    # return predictions
    return predictions


def feature_derivative(errors, feature):     
    # Compute the dot product of errors and feature
    derivative = np.dot(errors,feature)
    return derivative

def compute_avg_log_likelihood(feature_matrix, sentiment, coefficients):

    indicator = (sentiment==+1)
    scores = np.dot(feature_matrix, coefficients)
    logexp = np.log(1. + np.exp(-scores))
    
    # Simple check to prevent overflow
    mask = np.isinf(logexp)
    logexp[mask] = -scores[mask]   
    
    lp = np.sum((indicator-1)*scores - logexp)/len(feature_matrix)   
    
    return lp

j = 1                        # Feature number
i = 10                       # Data point number
coefficients = np.zeros(194) # A point w at which we are computing the gradient.

predictions = predict_probability(feature_matrix_train[i:i+1,:], coefficients)
indicator = (sentiment_train[i:i+1]==+1)

errors = indicator - predictions
gradient_single_data_point = feature_derivative(errors, feature_matrix_train[i:i+1,j])
print "Gradient single data point: %s" % gradient_single_data_point
print "           --> Should print 0.0"


j = 1                        # Feature number
i = 10                       # Data point start
B = 10                       # Mini-batch size
coefficients = np.zeros(194) # A point w at which we are computing the gradient.

predictions = predict_probability(feature_matrix_train[i:i+B,:], coefficients)
indicator = (sentiment_train[i:i+B]==+1)

errors = indicator - predictions
gradient_mini_batch = feature_derivative(errors, feature_matrix_train[i:i+B,j])
print "Gradient mini-batch data points: %s" % gradient_mini_batch
print "                --> Should print 1.0"


def logistic_regression_SG(feature_matrix, sentiment, initial_coefficients, step_size, batch_size, max_iter):
    log_likelihood_all = []

    # make sure it's a numpy array
    coefficients = np.array(initial_coefficients)
    # set seed=1 to produce consistent results
    np.random.seed(seed=1)
    # Shuffle the data before starting
    permutation = np.random.permutation(len(feature_matrix))
    feature_matrix = feature_matrix[permutation,:]
    sentiment = sentiment[permutation]

    i = 0 # index of current batch
    # Do a linear scan over data
    for itr in xrange(max_iter):
        # Predict P(y_i = +1|x_i,w) using your predict_probability() function
        # Make sure to slice the i-th row of feature_matrix with [i:i+batch_size,:]
        predictions = predict_probability(feature_matrix[i:i+batch_size,:], coefficients)

        # Compute indicator value for (y_i = +1)
        # Make sure to slice the i-th entry with [i:i+batch_size]
        indicator = (sentiment[i:i+batch_size]==+1)

        # Compute the errors as indicator - predictions
        errors = indicator - predictions
        for j in xrange(len(coefficients)): # loop over each coefficient
            # Recall that feature_matrix[:,j] is the feature column associated with coefficients[j]
            # Compute the derivative for coefficients[j] and save it to derivative.
            # Make sure to slice the i-th row of feature_matrix with [i:i+batch_size,j]
            derivative = feature_derivative(errors, feature_matrix[i:i+batch_size,j])
                  # Compute the product of the step size, the derivative, and
            # the **normalization constant** (1./batch_size)
            coefficients[j] += step_size*derivative*(1./batch_size)

        # Checking whether log likelihood is increasing
        # Print the log likelihood over the *current batch*
        lp = compute_avg_log_likelihood(feature_matrix[i:i+batch_size,:], sentiment[i:i+batch_size],
                                        coefficients)
        log_likelihood_all.append(lp)
        if itr <= 15 or (itr <= 1000 and itr % 100 == 0) or (itr <= 10000 and itr % 1000 == 0) \
         or itr % 10000 == 0 or itr == max_iter-1:
            data_size = len(feature_matrix)
            print 'Iteration %*d: Average log likelihood (of data points  [%0*d:%0*d]) = %.8f' % \
                (int(np.ceil(np.log10(max_iter))), itr, \
                 int(np.ceil(np.log10(data_size))), i, \
                 int(np.ceil(np.log10(data_size))), i+batch_size, lp)  

        # if we made a complete pass over data, shuffle and restart
        i += batch_size
        if i+batch_size > len(feature_matrix):
            permutation = np.random.permutation(len(feature_matrix))
            feature_matrix = feature_matrix[permutation,:]
            sentiment = sentiment[permutation]
            i = 0                

    # We return the list of log likelihoods for plotting purposes.
    return coefficients, log_likelihood_all

initial_coefficients = np.zeros(194)
step_size = 5e-1
batch_size = 1
max_iter = 10

log_batch1 = logistic_regression_SG(feature_matrix_train, sentiment_train, 
                       initial_coefficients, step_size, 
                       batch_size, max_iter)


initial_coefficients = np.zeros(194)
step_size = 5e-1
batch_size = len(feature_matrix_train)
max_iter = 200
    
log_batch_n = logistic_regression_SG(feature_matrix_train, sentiment_train, 
                       initial_coefficients, step_size, 
                       batch_size, max_iter)

step_size=1e-1
batch_size=100
initial_coefficients=np.zeros(194)

log_batch_100 = logistic_regression_SG(feature_matrix_train, sentiment_train, 
                       initial_coefficients, step_size, 
                       batch_size, max_iter)


import matplotlib.pyplot as plt

def make_plot(log_likelihood_all, len_data, batch_size, smoothing_window=1, label=''):
    plt.rcParams.update({'figure.figsize': (9,5)})
    log_likelihood_all_ma = np.convolve(np.array(log_likelihood_all), \
                                        np.ones((smoothing_window,))/smoothing_window, mode='valid')

    plt.plot(np.array(range(smoothing_window-1, len(log_likelihood_all)))*float(batch_size)/len_data,
             log_likelihood_all_ma, linewidth=4.0, label=label)
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()
    plt.xlabel('# of passes over data')
    plt.ylabel('Average log likelihood per data point')
    plt.legend(loc='lower right', prop={'size':14})

initial_coefficients = np.zeros(194)
step_size = 0.1
batch_size = 100
max_iter = 200
    
stochastic = logistic_regression_SG(feature_matrix_train, sentiment_train, 
                       initial_coefficients, step_size, 
                       batch_size, max_iter)

initial_coefficients = np.zeros(194)
step_size = 0.5
batch_size = len(feature_matrix_train)
max_iter = 200
    
batch_gradient = logistic_regression_SG(feature_matrix_train, sentiment_train, 
                       initial_coefficients, step_size, 
                       batch_size, max_iter)

initial_coefficients = np.zeros(194)
step_size = 1e-1
batch_size = 100
max_iter = 200

stochastic2 = logistic_regression_SG(feature_matrix_train, sentiment_train, 
                       initial_coefficients, step_size, 
                       batch_size, max_iter)

make_plot(batch_gradient[1], len(feature_matrix_train), len(feature_matrix_train), 30, '')

make_plot(stochastic[1], len(feature_matrix_train), 100, 30, '')

make_plot(stochastic2[1], len(feature_matrix_train), 100, 30, '')    

# Iterate over step sizes
step_sizes = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]

initial_coefficients = np.zeros(194)
batch_size = 100
max_iter = 10

logs = {}
for step_size in step_sizes:
   logs[step_size] = logistic_regression_SG(feature_matrix_train, sentiment_train, 
                       initial_coefficients, step_size, 
                       batch_size, max_iter)[1]
   

make_plot(logs[1e-4], len(feature_matrix_train), step_size, 30, '')
make_plot(logs[1e-3], len(feature_matrix_train), step_size, 30, '')
make_plot(logs[1e-2], len(feature_matrix_train), step_size, 30, '')
make_plot(logs[1e-1], len(feature_matrix_train), step_size, 30, '')
make_plot(logs[1e0], len(feature_matrix_train), step_size, 30, '')
make_plot(logs[1e1], len(feature_matrix_train), step_size, 30, '')
make_plot(logs[1e2], len(feature_matrix_train), step_size, 30, '')
