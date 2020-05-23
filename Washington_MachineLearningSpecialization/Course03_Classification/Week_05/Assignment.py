import os
path = 'C:/Users/SvenV/Desktop/Coursera Kurse/Machine Learning Spezialization III/Assignments/Week 5'
os.chdir(path)
os.listdir(path)

import pandas as pd

loans = pd.read_csv('lending-club-data.csv')

loans.columns
loans.head(2)

loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.drop('bad_loans',axis=1)

target = 'safe_loans'
features = ['grade',                     # grade of the loan (categorical)
            'sub_grade_num',             # sub-grade of the loan as a number from 0 to 1
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'payment_inc_ratio',         # ratio of the monthly payment to income
            'delinq_2yrs',               # number of delinquincies
             'delinq_2yrs_zero',          # no delinquincies in last 2 years
            'inq_last_6mths',            # number of creditor inquiries in last 6 months
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'open_acc',                  # number of open credit accounts
            'pub_rec',                   # number of derogatory public records
            'pub_rec_zero',              # no derogatory public records
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
            'int_rate',                  # interest rate of the loan
            'total_rec_int',             # interest received to date
            'annual_inc',                # annual income of borrower
            'funded_amnt',               # amount committed to the loan
            'funded_amnt_inv',           # amount committed by investors for the loan
            'installment',               # monthly payment owed by the borrower
           ]

loans = loans[[target] + features].dropna()

# Encode data for model
dummies = pd.get_dummies(loans.select_dtypes(include=[object]))

all_data = loans.select_dtypes(exclude=[object]).join(dummies)

# Split into Training and Validation
train_data = all_data.iloc[pd.read_json('module-8-assignment-1-train-idx.json')[0],:]
validation_data = all_data.iloc[pd.read_json('module-8-assignment-1-validation-idx.json')[0],:]

safe_loans_raw = loans[loans[target] == 1]
risky_loans_raw = loans[loans[target] == -1]

import sklearn
from sklearn import ensemble
import numpy as np

validation_safe_loans = validation_data[validation_data[target] == 1]
validation_risky_loans = validation_data[validation_data[target] == -1]

sample_validation_data_risky = validation_risky_loans[0:2]
sample_validation_data_safe = validation_safe_loans[0:2]

sample_validation_data = sample_validation_data_safe.append(sample_validation_data_risky)
sample_validation_data

# Import functions for decision tree
from functions_decision_tree import *

model_5 = decision_tree_create(train_data, list(train_data.columns[1:]), 'safe_loans', 0, max_depth = 6, min_node_size = 0, min_error_reduction=0)

for i in xrange(len(sample_validation_data)):
    print "Case: " + str(i) + " Prediction: " + str(classify(model_5, sample_validation_data.iloc[i], annotate = False))

sample_validation_data[target]

# With sklearn
gbes = ensemble.GradientBoostingClassifier(n_estimators=5,max_depth=6)
model5 = gbes.fit(train_data.drop([target],axis=1), train_data[target])

predictions = model5.predict(sample_validation_data.drop([target],axis=1)) 

predictions == sample_validation_data[target]

predictions_p = model5.predict_proba(sample_validation_data.drop([target],axis=1))[:,1]

np.column_stack((predictions,predictions_p))

from sklearn.metrics import accuracy_score
accuracy_score(validation_data[target],model5.predict(validation_data.drop([target],axis=1)))

# false positives: prediction 1, actual -1 -> 0
predictions = pd.concat([pd.Series(validation_data[target],name='true').reset_index(),pd.Series(model5.predict(validation_data.drop([target],axis=1)),name='predict')],axis=1)
false_positives = predictions.loc[(predictions['true']==-1) & (predictions['predict']==+1)].count()[0]
false_negatives = predictions.loc[(predictions['true']==1) & (predictions['predict']==-1)].count()[0]

cost = 10000 * false_negatives  + 20000 * false_positives

validation_data.loc[:,'Prob_Safe'] = pd.Series(model5.predict_proba(validation_data.drop([target],axis=1))[:,1], index=validation_data.index)

validation_data_desc = validation_data.sort_values(by=['Prob_Safe'],axis=0,ascending=False)
validation_data_asc = validation_data.sort_values(by=['Prob_Safe'],axis=0,ascending=True)

validation_data_desc[[col for col in validation_data_desc.columns if 'grade' in col]].head(5)

validation_data_asc[[col for col in validation_data_asc.columns if 'grade' in col]].head(5)

# Different models
gbes10  = ensemble.GradientBoostingClassifier(n_estimators=10,max_depth=6)
model10 = gbes10.fit(train_data.drop([target],axis=1), train_data[target])
gbes50 = ensemble.GradientBoostingClassifier(n_estimators=50,max_depth=6)
model50 = gbes50.fit(train_data.drop([target],axis=1), train_data[target])
gbes100 = ensemble.GradientBoostingClassifier(n_estimators=100,max_depth=6)
model100 = gbes100.fit(train_data.drop([target],axis=1), train_data[target])
gbes200 = ensemble.GradientBoostingClassifier(n_estimators=200,max_depth=6)
model200 = gbes200.fit(train_data.drop([target],axis=1), train_data[target])
gbes500 = ensemble.GradientBoostingClassifier(n_estimators=500,max_depth=6)
model500 = gbes500.fit(train_data.drop([target],axis=1), train_data[target])

acc10 = accuracy_score(validation_data[target],model10.predict(validation_data.drop([target,'Prob_Safe'],axis=1)))
acc50 = accuracy_score(validation_data[target],model50.predict(validation_data.drop([target,'Prob_Safe'],axis=1)))
acc100 = accuracy_score(validation_data[target],model100.predict(validation_data.drop([target,'Prob_Safe'],axis=1)))
acc200 = accuracy_score(validation_data[target],model200.predict(validation_data.drop([target,'Prob_Safe'],axis=1)))
acc500 = accuracy_score(validation_data[target],model500.predict(validation_data.drop([target,'Prob_Safe'],axis=1)))

acc10, acc50, acc100, acc200, acc500

import matplotlib.pyplot as plt
%matplotlib inline
def make_figure(dim, title, xlabel, ylabel, legend):
    plt.rcParams['figure.figsize'] = dim
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend is not None:
        plt.legend(loc=legend, prop={'size':15})
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()
    
acc10_train = accuracy_score(train_data[target],model10.predict(train_data.drop([target],axis=1)))
acc50_train = accuracy_score(train_data[target],model50.predict(train_data.drop([target],axis=1)))
acc100_train = accuracy_score(train_data[target],model100.predict(train_data.drop([target],axis=1)))
acc200_train = accuracy_score(train_data[target],model200.predict(train_data.drop([target],axis=1)))
acc500_train = accuracy_score(train_data[target],model500.predict(train_data.drop([target],axis=1)))

training_errors = [1-acc10_train, 1-acc50_train, 1-acc100_train, 1-acc200_train, 1-acc500_train]

validation_errors = [1-acc10, 1-acc50, 1-acc100, 1-acc200, 1-acc500]

plt.plot([10, 50, 100, 200, 500], training_errors, linewidth=4.0, label='Training error')
plt.plot([10, 50, 100, 200, 500], validation_errors, linewidth=4.0, label='Validation error')

make_figure(dim=(10,5), title='Error vs number of trees',
            xlabel='Number of trees',
            ylabel='Classification error',
            legend='best')