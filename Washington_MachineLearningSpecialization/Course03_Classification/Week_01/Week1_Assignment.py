# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 21:42:21 2017

@author: SvenV
"""

import os
path = 'C:/Users/SvenV/Desktop/Coursera Kurse/Machine Learning Spezialization III/Assignments'
os.chdir(path)
os.listdir(path)

import pandas as pd
import numpy as np

products = pd.read_csv('amazon_baby.csv')

products.iloc[0]
products.columns

# Fill all NAs with empty blanks
products[products['review'].isnull()]
products = products.fillna({'review':''})
products[products['review'].isnull()]

import string
products['review_clean'] = products['review'].str.replace('[{}]'.format(string.punctuation), '') 

products.head(1)

products.iloc[166744]

# Exclude rating equals 3
products = products[products['rating'] != 3]

products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)
products['sentiment'].value_counts()

train_data = products.iloc[pd.read_json('module-2-assignment-train-idx.json')[0],:]
test_data = products.iloc[pd.read_json('module-2-assignment-test-idx.json')[0],:]

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
# Use this token pattern to keep single-letter words
# First, learn vocabulary from the training data and assign columns to words
# Then convert the training data into a sparse matrix
train_matrix = vectorizer.fit_transform(train_data['review_clean'].values.astype('U'))
# Second, convert the test data into a sparse matrix, using the same word-column mapping
test_matrix = vectorizer.transform(test_data['review_clean'].values.astype('U'))

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
sentiment_model = model.fit(train_matrix, train_data['sentiment'])

sentiment_model.coef_[sentiment_model.coef_>=0].size

sample_test_data = test_data[10:13]
print sample_test_data

sample_test_data.iloc[0]
sample_test_data.iloc[1]

sample_test_matrix = vectorizer.transform(sample_test_data['review_clean'])
scores = sentiment_model.decision_function(sample_test_matrix)
print scores

1/(1+np.exp(-scores))

# Predict Probability
test_data.loc[:,'p'] = pd.Series(sentiment_model.predict_proba(test_matrix)[:,1], index=test_data.index)

top20 = test_data.sort('p',ascending=False)[0:19]
bottom20 = test_data.sort('p')[0:19]

'Safety 1st Exchangeable Tip 3 in 1 Thermometer' in list(top20['name'])
'The First Years True Choice P400 Premium Digital Monitor, 2 Parent Unit' in list(bottom20['name'])

# Predict class
test_data.loc[:,'predict_class'] = pd.Series(sentiment_model.predict(test_matrix), index=test_data.index)

float(sum(test_data['sentiment'] == test_data['predict_class']))/test_data.shape[0]

### Only 20 words
significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves', 
      'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed', 
      'work', 'product', 'money', 'would', 'return']

vectorizer_word_subset = CountVectorizer(vocabulary=significant_words) # limit to 20 words
train_matrix_word_subset = vectorizer_word_subset.fit_transform(train_data['review_clean'])
test_matrix_word_subset = vectorizer_word_subset.transform(test_data['review_clean'])

simple_model = model.fit(train_matrix_word_subset, train_data['sentiment'])

coef_dict = {}
for coef, feat in zip(simple_model.coef_[0],significant_words):
    coef_dict[feat] = coef
    
sentiment_model.coef_[0]
train_matrix

train_data.loc[:,'predict_class'] = pd.Series(sentiment_model.predict(train_matrix), index=train_data.index)
train_data.loc[:,'predict_class_simple'] = pd.Series(simple_model.predict(train_matrix_word_subset), index=train_data.index)

float(sum(train_data['sentiment'] == train_data['predict_class']))/train_data.shape[0]
float(sum(train_data['sentiment'] == train_data['predict_class_simple']))/train_data.shape[0]


test_data.loc[:,'predict_class'] = pd.Series(sentiment_model.predict(test_matrix), index=test_data.index)
test_data.loc[:,'predict_class_simple'] = pd.Series(simple_model.predict(test_matrix_word_subset), index=test_data.index)

float(sum(test_data['sentiment'] == test_data['predict_class']))/test_data.shape[0]
float(sum(test_data['sentiment'] == test_data['predict_class_simple']))/test_data.shape[0]

from scipy.stats import mode
float(sum(test_data['sentiment'] == mode(test_data['sentiment'])[0][0]))/test_data.shape[0]
