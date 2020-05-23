# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 08:19:05 2018

@author: SvenV
"""

import os
path = 'C:/Users/SvenV/Desktop/Coursera Kurse/Machine Learning Spezialization III/Assignments/Week 6'
os.chdir(path)
os.listdir(path)

import pandas as pd
import numpy as np

products = pd.read_csv('amazon_baby.csv')
products.iloc[0]

import string
products['review_clean'] = products['review'].str.replace('[{}]'.format(string.punctuation), '')

products = products[products['rating'] != 3]

products['sentiment'] = np.where(products['rating'] > 3, 1, -1)

# Split into Training and Validation
train_data = products.iloc[pd.read_json('module-9-assignment-train-idx.json')[0],:]
test_data = products.iloc[pd.read_json('module-9-assignment-test-idx.json')[0],:]

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
# Use this token pattern to keep single-letter words
# First, learn vocabulary from the training data and assign columns to words
# Then convert the training data into a sparse matrix
train_matrix = vectorizer.fit_transform(train_data['review_clean'].values.astype('U'))
# Second, convert the test data into a sparse matrix, using the same word-column mapping
test_matrix = vectorizer.transform(test_data['review_clean'].values.astype('U'))

from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()
model = logistic.fit(train_matrix, train_data['sentiment'])

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true=test_data['sentiment'], y_pred=model.predict(test_matrix))
print "Test Accuracy: %s" % accuracy

baseline = float(len(test_data[test_data['sentiment'] == 1]))/len(test_data)
print "Baseline accuracy (majority class classifier): %s" % baseline

temp = test_data

temp.loc[:,'f'] = pd.Series(test_data['sentiment'] == model.predict(test_matrix),
        index=temp.index)

len(temp[(temp['sentiment'] == -1) & (temp['f'] == False)])
len(temp[(temp['sentiment'] == -1) & (temp['f'] == False)])/float(len(temp[(temp['sentiment'] == 1)]))

### Confusion Matrix
from sklearn.metrics import confusion_matrix
cmat = confusion_matrix(y_true=test_data['sentiment'],
                        y_pred=model.predict(test_matrix),
                        labels=model.classes_)    # use the same order of class as the LR model.
print ' target_label | predicted_label | count '
print '--------------+-----------------+-------'
# Print out the confusion matrix.
# NOTE: Your tool may arrange entries in a different order. Consult appropriate manuals.
for i, target_label in enumerate(model.classes_):
    for j, predicted_label in enumerate(model.classes_):
        print '{0:^13} | {1:^15} | {2:5d}'.format(target_label, predicted_label, cmat[i,j])
        
false_positives = 1444 
false_negatives = 813     
100*false_positives + 1*false_negatives

from sklearn.metrics import precision_score
precision = precision_score(y_true=test_data['sentiment'], 
                            y_pred=model.predict(test_matrix))
print "Precision on test data: %s" % precision

float(813)/(813+27282)

from sklearn.metrics import recall_score
recall = recall_score(y_true=test_data['sentiment'],
                      y_pred=model.predict(test_matrix))
print "Recall on test data: %s" % recall


def apply_treshold(probabilities,treshold):
    return pd.Series(np.where(probabilities > treshold, 1, -1))

probabilities = model.predict_proba(test_matrix)[:,1]

apply_treshold(probabilities,0.5).value_counts(normalize=True)
apply_treshold(probabilities,0.9).value_counts(normalize=True)

precision_05 = precision_score(y_true=test_data['sentiment'], y_pred=apply_treshold(probabilities,0.5))
precision_09 = precision_score(y_true=test_data['sentiment'], y_pred=apply_treshold(probabilities,0.9))

recall_05 = recall_score(y_true=test_data['sentiment'], y_pred=apply_treshold(probabilities,0.5))
recall_09 = recall_score(y_true=test_data['sentiment'], y_pred=apply_treshold(probabilities,0.9))

threshold_values = np.linspace(0.5, 0.99, num=100)
print threshold_values

precision_all = []
recall_all = []
for thresh in threshold_values:
    precision_all.append(precision_score(y_true=test_data['sentiment'], y_pred=apply_treshold(probabilities,thresh)))
    recall_all.append(recall_score(y_true=test_data['sentiment'], y_pred=apply_treshold(probabilities,thresh)))

import matplotlib.pyplot as plt

def plot_pr_curve(precision, recall, title):
    plt.rcParams['figure.figsize'] = 7, 5
    plt.locator_params(axis = 'x', nbins = 5)
    plt.plot(precision, recall, 'b-', linewidth=4.0, color = '#B0017F')
    plt.title(title)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.rcParams.update({'font.size': 16})
    
plot_pr_curve(precision_all, recall_all, 'Precision recall curve (all)')

data = pd.DataFrame({'Thresh': threshold_values,'Precision': precision_all,'Recall': recall_all})

data[data['Precision'] >= 0.965].iloc[0]

recall_score(y_true=test_data['sentiment'], y_pred=
             apply_treshold(probabilities,0.98))*sum(test_data['sentiment']==1)

temp = test_data

temp.loc[:,'f'] = pd.Series(test_data['sentiment'] == np.array(apply_treshold(probabilities,0.98)),
        index=temp.index)

len(temp[(temp['sentiment'] == 1) & (temp['f'] == False)])
len(temp[(temp['sentiment'] == -1) & (temp['f'] == False)])/float(len(temp[(temp['sentiment'] == 1)]))


baby_reviews = test_data[test_data['name'].str.contains('baby',case=False) == True].dropna(axis=0, how='any')

baby_matrix = vectorizer.transform(baby_reviews['review_clean'])
probabilities_babies = model.predict_proba(baby_matrix)[:,1]

precision_babys = []
recall_babys = []
for thresh in threshold_values:
    precision_babys.append(precision_score(y_true=baby_reviews['sentiment'], y_pred=apply_treshold(probabilities_babies,thresh)))
    recall_babys.append(recall_score(y_true=baby_reviews['sentiment'], y_pred=apply_treshold(probabilities_babies,thresh)))

data_babys = pd.DataFrame({'Thresh': threshold_values,'Precision': precision_babys,'Recall': recall_babys})

data_babys[data_babys['Precision'] >= 0.965].iloc[0]

plot_pr_curve(precision_babys, recall_babys, "Precision-Recall (Baby)")

