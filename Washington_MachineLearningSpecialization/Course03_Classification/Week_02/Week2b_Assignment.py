# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 21:34:09 2017

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
    print(word)
    products[word] = products['review_clean'].apply(lambda s : s.split().count(word))
    
# Train Validation Split
train_data = products.iloc[pd.read_json('module-4-assignment-train-idx.json')[0],:]
validation_data = products.iloc[pd.read_json('module-4-assignment-validation-idx.json')[0],:]

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


def feature_derivative_with_L2(errors, feature, coefficient,l2_penalty, feature_is_constant):     
    # Compute the dot product of errors and feature
    derivative = np.dot(errors,feature)
    # Add L2 penalty term
    if not feature_is_constant:
        derivative = derivative-2*np.dot(l2_penalty,coefficient)
    return derivative


def compute_log_likelihood_with_L2(feature_matrix, sentiment, coefficients, l2_penalty):
    indicator = (sentiment==+1)
    scores = np.dot(feature_matrix, coefficients)
    
    lp = np.sum((indicator-1)*scores - np.log(1. + np.exp(-scores))) - l2_penalty*np.sum(coefficients[1:]**2)
    
    return lp


def logistic_regression_with_L2(feature_matrix, sentiment, initial_coefficients, step_size, l2_penalty, max_iter):
    coefficients = np.array(initial_coefficients) # make sure it's a numpy array
    for itr in xrange(max_iter):
        predictions = predict_probability(feature_matrix, coefficients)
        indicator = (sentiment==+1)
        errors = indicator - predictions

        for j in xrange(len(coefficients)): 
            is_intercept = (j == 0)
            #print "Iteration " + str(itr) + " updating weight " + str(j)
            derivative = feature_derivative_with_L2(errors, feature_matrix[:,j],coefficients[j], l2_penalty, is_intercept)
            coefficients[j] = coefficients[j] + np.dot(step_size,derivative)

        # Checking whether log likelihood is increasing
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
        or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            lp = compute_log_likelihood_with_L2(feature_matrix, sentiment, coefficients, l2_penalty)
            print 'iteration %*d: log likelihood of observed labels = %.8f' % \
                (int(np.ceil(np.log10(max_iter))), itr, lp)
    return coefficients   


feature_matrix = feature_matrix_train
sentiment = sentiment_train
initial_coefficients = np.repeat(float(0),feature_matrix.shape[1])
step_size = 5e-6
max_iter = 501

l2_penalty_list = [0,4,10,1e2,1e3,1e5]

coefficients_penalty = dict()
coefficients_penalty_wI = dict()
for l2_penalty in l2_penalty_list:
    coefficients_penalty_wI[l2_penalty] = logistic_regression_with_L2(feature_matrix, sentiment, initial_coefficients, step_size, l2_penalty, max_iter)
    coefficients_penalty[l2_penalty] = pd.DataFrame({'words': important_words, 'coef_' + str(l2_penalty): coefficients_penalty_wI[l2_penalty][1:]})

# get five most positive and negative words with 0 penalty
positive_words = list(coefficients_penalty[0].nlargest(5, 'coef_0')['words'])
negative_words = list(coefficients_penalty[0].nsmallest(5, 'coef_0')['words'])

# Subset and combine all pandas to only these words
table = pd.DataFrame({'words': positive_words + negative_words})
for l2_penalty in l2_penalty_list:
    table['coef_' + str(l2_penalty)] = list(coefficients_penalty[l2_penalty][coefficients_penalty[l2_penalty]['words'].isin(positive_words + negative_words)]['coef_' + str(l2_penalty)])

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 10, 6

def make_coefficient_plot(table, positive_words, negative_words, l2_penalty_list):
    cmap_positive = plt.get_cmap('Reds')
    cmap_negative = plt.get_cmap('Blues')
    
    xx = l2_penalty_list
    plt.plot(xx, [0.]*len(xx), '--', lw=1, color='k')
    
    table_positive_words = table[table['words'].isin(positive_words)]
    table_negative_words = table[table['words'].isin(negative_words)]
    
    del table_positive_words['words']
    del table_negative_words['words']
    
    for i in xrange(len(positive_words)):
        color = cmap_positive(0.8*((i+1)/(len(positive_words)*1.2)+0.15))
        plt.plot(xx, table_positive_words[i:i+1].as_matrix().flatten(),
                 '-', label=positive_words[i], linewidth=4.0, color=color)
        
    for i in xrange(len(negative_words)):
        color = cmap_negative(0.8*((i+1)/(len(negative_words)*1.2)+0.15))
        plt.plot(xx, table_negative_words[i:i+1].as_matrix().flatten(),
                 '-', label=negative_words[i], linewidth=4.0, color=color)
        
    plt.legend(loc='best', ncol=3, prop={'size':16}, columnspacing=0.5)
    plt.axis([1, 1e5, -1, 2])
    plt.title('Coefficient path')
    plt.xlabel('L2 penalty ($\lambda$)')
    plt.ylabel('Coefficient value')
    plt.xscale('log')
    plt.rcParams.update({'font.size': 18})
    plt.tight_layout()

make_coefficient_plot(table, positive_words, negative_words, l2_penalty_list=[0, 4, 10, 1e2, 1e3, 1e5])


scores_train = dict()
scores_valid = dict()
accuracy_train = dict()
accuracy_valid = dict()
for l2_penalty in l2_penalty_list:
    scores_train[l2_penalty] = np.dot(feature_matrix_train,coefficients_penalty_wI[l2_penalty])
    scores_valid[l2_penalty] = np.dot(feature_matrix_valid,coefficients_penalty_wI[l2_penalty])
    accuracy_train[l2_penalty] = pd.Series(sentiment_train == pd.Series(np.where(scores_train[l2_penalty] > 0, 1, -1))).value_counts()[1]/float(len(sentiment_train))
    accuracy_valid[l2_penalty] = pd.Series(sentiment_valid == pd.Series(np.where(scores_valid[l2_penalty] > 0, 1, -1))).value_counts()[1]/float(len(sentiment_valid))

accuracy_train
accuracy_valid


