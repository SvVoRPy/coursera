import os
path = 'C:/Users/SvenV/Desktop/Coursera Kurse/Machine Learning Spezialization II/Assignments'
os.chdir(path)
os.listdir(path)

import pandas as pd
import numpy as np

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float,
              'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str,
              'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int,
              'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str,
              'sqft_lot':int, 'view':int}

sales = pd.read_csv('kc_house_data.csv',dtype=dtype_dict)
test_data = pd.read_csv('kc_house_test_data.csv',dtype=dtype_dict)
train_data = pd.read_csv('kc_house_train_data.csv',dtype=dtype_dict)

# Create new features by performing following transformation on inputs:
from math import log, sqrt
sales['sqft_living_sqrt'] = sales['sqft_living'].apply(sqrt)
sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(sqrt)
sales['bedrooms_square'] = sales['bedrooms']*sales['bedrooms']
sales['floors_square'] = sales['floors']*sales['floors']

### Learn the Model
from sklearn import linear_model  # using scikit-learn

all_features = ['bedrooms', 'bedrooms_square',
            'bathrooms',
            'sqft_living', 'sqft_living_sqrt',
            'sqft_lot', 'sqft_lot_sqrt',
            'floors', 'floors_square',
            'waterfront', 'view', 'condition', 'grade',
            'sqft_above',
            'sqft_basement',
            'yr_built', 'yr_renovated']

model_all = linear_model.Lasso(alpha=5e2, normalize=True) # set parameters
model_all.fit(sales[all_features], sales['price']) # learn weights

coef_dict = {}
for coef, feat in zip(model_all.coef_,all_features):
    if coef!=0:
        coef_dict[feat] = coef

# Koeffizienten ungleich Null für:
coef_dict


# Unterteilung Daten in Testing, Training und Validation
testing = pd.read_csv('wk3_kc_house_test_data.csv', dtype=dtype_dict)
training = pd.read_csv('wk3_kc_house_train_data.csv', dtype=dtype_dict)
validation = pd.read_csv('wk3_kc_house_valid_data.csv', dtype=dtype_dict)

testing['sqft_living_sqrt'] = testing['sqft_living'].apply(sqrt)
testing['sqft_lot_sqrt'] = testing['sqft_lot'].apply(sqrt)
testing['bedrooms_square'] = testing['bedrooms']*testing['bedrooms']
testing['floors_square'] = testing['floors']*testing['floors']

training['sqft_living_sqrt'] = training['sqft_living'].apply(sqrt)
training['sqft_lot_sqrt'] = training['sqft_lot'].apply(sqrt)
training['bedrooms_square'] = training['bedrooms']*training['bedrooms']
training['floors_square'] = training['floors']*training['floors']

validation['sqft_living_sqrt'] = validation['sqft_living'].apply(sqrt)
validation['sqft_lot_sqrt'] = validation['sqft_lot'].apply(sqrt)
validation['bedrooms_square'] = validation['bedrooms']*validation['bedrooms']
validation['floors_square'] = validation['floors']*validation['floors']


# Test verschiedener L1 Penalties
penalties = np.logspace(1, 7, num=13)
model = {}
RSS = {}
for l1 in penalties:
    model[l1] = linear_model.Lasso(alpha=l1,normalize=True).fit(
            training[all_features], training['price'])
    RSS[l1] = sum((validation['price'] - model[l1].predict(validation[all_features]))**2)
    
l1 = min(RSS, key=RSS.get)

# Test data RSS with best penalty
sum((testing['price'] - model[l1].predict(testing[all_features]))**2)

# Wie viele Nonzeros im Modell?
coef_dict = {}
for coef, feat in zip(model[l1].coef_,all_features):
    if coef!=0:
        coef_dict[feat] = coef

if model[l1].intercept_!=0:
    coef_dict['_intercept'] = model[l1].intercept_

len(coef_dict)


### Ziel: Modell soll höchstens sieben Features enthalten
penalties = np.logspace(1, 4, num=20)
model = {}
n_coef = {}
for l1 in penalties:
    model[l1] = linear_model.Lasso(alpha=l1,normalize=True).fit(
            training[all_features], training['price'])
    n_coef[l1] = np.sum(model[l1].coef_!=0)
    if model[l1].intercept_ != 0:
        n_coef[l1] = n_coef[l1] + 1
        
max_nonzeros = 7        

l1_penalty_min = max({k:v for (k,v) in n_coef.items() if v > 7})
l1_penalty_max = min({k:v for (k,v) in n_coef.items() if v < 7})

RSS_7 = {}

for l1 in np.linspace(l1_penalty_min,l1_penalty_max,20):
    model[l1] = linear_model.Lasso(alpha=l1,normalize=True).fit(
            training[all_features], training['price'])
    n_coef[l1] = np.sum(model[l1].coef_!=0)
    if model[l1].intercept_ != 0:
        n_coef[l1] = n_coef[l1] + 1
    if n_coef[l1] == 7:
        RSS_7[l1] = sum((validation['price'] - model[l1].predict(validation[all_features]))**2)        

l1 = min(RSS_7, key=RSS_7.get)

coef_dict = {}
for coef, feat in zip(model[l1].coef_,all_features):
    if coef!=0:
        coef_dict[feat] = coef

if model[l1].intercept_!=0:
    coef_dict['_intercept'] = model[l1].intercept_
        
coef_dict
