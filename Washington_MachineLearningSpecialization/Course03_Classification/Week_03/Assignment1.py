# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 23:04:54 2017

@author: SvenV
"""

import os
path = 'C:/Users/SvenV/Desktop/Coursera Kurse/Machine Learning Spezialization III/Assignments/Week 3'
os.chdir(path)
os.listdir(path)

import pandas as pd
import numpy as np

loans = pd.read_csv('lending-club-data.csv')
loans.columns
loans.head(2)

loans['bad_loans'].value_counts()
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
del loans['bad_loans']

loans['safe_loans'].value_counts()
loans['safe_loans'].value_counts(normalize=True)

features = ['grade',                     # grade of the loan
            'sub_grade',                 # sub-grade of the loan
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'term',                      # the term of the loan
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
           ]

target = 'safe_loans'                    # prediction target (y) (+1 means safe, -1 is risky)

# Extract the feature columns and target column
loans = loans[features + [target]]

# Encode data for model
dummies = pd.get_dummies(loans.select_dtypes(include=[object]))

all_data = loans.select_dtypes(exclude=[object]).join(dummies)

# Split into Training and Validation
train_data = all_data.iloc[pd.read_json('module-5-assignment-1-train-idx.json')[0],:]
validation_data = all_data.iloc[pd.read_json('module-5-assignment-1-validation-idx.json')[0],:]

# Get check data
safe_loans_raw = loans[loans[target] == +1]
risky_loans_raw = loans[loans[target] == -1]
print "Number of safe loans  : %s" % len(safe_loans_raw)
print "Number of risky loans : %s" % len(risky_loans_raw)

validation_safe_loans = validation_data[validation_data[target] == 1]
validation_risky_loans = validation_data[validation_data[target] == -1]

sample_validation_data_risky = validation_risky_loans[0:2]
sample_validation_data_safe = validation_safe_loans[0:2]

sample_validation_data = sample_validation_data_safe.append(sample_validation_data_risky)

# Tree prediction
from sklearn.tree import DecisionTreeClassifier

decision_tree_model = DecisionTreeClassifier(max_depth=6)
small_model = DecisionTreeClassifier(max_depth=2)

decision_tree_model = decision_tree_model.fit(train_data.drop('safe_loans',axis=1), train_data['safe_loans'])
small_model = small_model.fit(validation_data.drop('safe_loans',axis=1), validation_data['safe_loans'])

n_nodes = decision_tree_model.tree_.node_count
n_nodes_small = small_model.tree_.node_count

sample_validation_data['safe_loans'] 
pd.Series(decision_tree_model.predict(sample_validation_data.drop('safe_loans',axis=1)))
(2/float(4))*100

decision_tree_model.predict_proba(sample_validation_data.drop('safe_loans',axis=1))

small_model.predict(sample_validation_data.drop('safe_loans',axis=1))
small_model.predict_proba(sample_validation_data.drop('safe_loans',axis=1))

import graphviz
from sklearn import tree
dot_data = tree.export_graphviz(small_model, out_file=None) 
graphviz.Source(dot_data) 

from sklearn.metrics import accuracy_score

accuracy_score(train_data['safe_loans'],decision_tree_model.predict(train_data.drop('safe_loans',axis=1)))
accuracy_score(train_data['safe_loans'],small_model.predict(train_data.drop('safe_loans',axis=1)))

accuracy_score(validation_data['safe_loans'],decision_tree_model.predict(validation_data.drop('safe_loans',axis=1)))
accuracy_score(validation_data['safe_loans'],small_model.predict(validation_data.drop('safe_loans',axis=1)))

big_model = DecisionTreeClassifier(max_depth=10)
big_model = big_model.fit(train_data.drop('safe_loans',axis=1), train_data['safe_loans'])

accuracy_score(train_data['safe_loans'],big_model.predict(train_data.drop('safe_loans',axis=1)))
accuracy_score(validation_data['safe_loans'],big_model.predict(validation_data.drop('safe_loans',axis=1)))

def cost_of_prediction(model,features,labels,cost_false_positive,cost_false_negative):
    predictions = pd.concat([pd.Series(labels,name='true').reset_index(),pd.Series(model.predict(features),name='predict')],axis=1)
    false_positives = predictions.loc[(predictions['true']==-1) & (predictions['predict']==+1)].count()[0]
    false_negatives = predictions.loc[(predictions['true']==1) & (predictions['predict']==-1)].count()[0]
    cost_mistakes = false_positives*cost_false_positive + false_negatives*cost_false_negative
    return cost_mistakes
    
cost_of_prediction(big_model,validation_data.drop('safe_loans',axis=1),
                   validation_data['safe_loans'],20000,10000)

