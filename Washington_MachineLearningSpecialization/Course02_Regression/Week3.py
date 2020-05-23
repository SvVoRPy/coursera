import os
path = 'C:/Users/SvenV/Desktop/Coursera Kurse/Machine Learning Spezialization II/Assignments'
os.chdir(path)
os.listdir(path)

import pandas as pd
import numpy as np

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

house_data = pd.read_csv('kc_house_data.csv',dtype=dtype_dict)
house_test_data = pd.read_csv('wk3_kc_house_test_data.csv',dtype=dtype_dict)
house_train_data = pd.read_csv('wk3_kc_house_train_data.csv',dtype=dtype_dict)
house_valid_data = pd.read_csv('wk3_kc_house_valid_data.csv',dtype=dtype_dict)

house_set1 = pd.read_csv('wk3_kc_house_set_1_data.csv',dtype=dtype_dict)
house_set2 = pd.read_csv('wk3_kc_house_set_2_data.csv',dtype=dtype_dict)
house_set3 = pd.read_csv('wk3_kc_house_set_3_data.csv',dtype=dtype_dict)
house_set4 = pd.read_csv('wk3_kc_house_set_4_data.csv',dtype=dtype_dict)


### Polynomial function

def polynomial_dataframe(feature, degree): # feature is pandas.Series type
    # assume that degree >= 1
    # initialize the dataframe:
    poly_dataframe = pd.DataFrame()
    # and set poly_dataframe['power_1'] equal to the passed feature
    poly_dataframe['power_1'] = feature
    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        for power in range(2, degree+1):
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # assign poly_dataframe[name] to be feature^power; use apply(*)
            poly_dataframe[name] = feature.apply(lambda x: x**power)
    return poly_dataframe

sales = house_data.sort_values(['sqft_living','price'])

poly1_data = polynomial_dataframe(sales['sqft_living'],1)

poly1_data['price'] = sales['price']

import statsmodels.api as sm
model1 = sm.OLS(poly1_data['price'],sm.add_constant(poly1_data['power_1'])).fit()

import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(poly1_data['power_1'],poly1_data['price'],'.',
         poly1_data['power_1'], model1.predict(),'-')

# 2nd and 3rd polynomial:
poly_data = polynomial_dataframe(sales['sqft_living'],3)   
poly_data['price'] = sales['price']
model2 = sm.OLS(poly_data['price'],sm.add_constant(poly_data[['power_1',
                'power_2']])).fit()
model3 = sm.OLS(poly_data['price'],sm.add_constant(poly_data[['power_1',
                'power_2','power_3']])).fit()
model2.params
model3.params

plt.plot(poly1_data['power_1'],poly1_data['price'],'.',
         poly_data['power_1'], model2.predict(),'-')
plt.plot(poly1_data['power_1'],poly1_data['price'],'.',
         poly_data['power_1'], model3.predict(),'-')

# 15th polynomial
poly15_data = polynomial_dataframe(sales['sqft_living'],15)   
poly15_data['price'] = sales['price']

model15 = sm.OLS(poly_data['price'],sm.add_constant(
        poly15_data.drop(['price'],axis=1))).fit()

model15.params
    
plt.plot(poly1_data['power_1'],poly1_data['price'],'.',
         poly_data['power_1'], model15.predict(),'-')

# Für alle vier separaten Datensätze
house_set = {1: house_set1 , 2: house_set2 , 3: house_set3 , 4: house_set4}
daten = {}
model = {}
for i in range(1,5):
    daten[i] = polynomial_dataframe(house_set[i]['sqft_living'],15)
    daten[i]['price'] = house_set[i]['price']
    model[i] = sm.OLS(daten[i]['price'],sm.add_constant(daten[i].drop(['price'],axis=1))).fit()

plt.subplots(2,2)
for i in range(1,5):
    ax=plt.subplot(2,2,i)
    ax.plot(daten[i]['power_1'], daten[i]['price'],'.',
         daten[i]['power_1'], model[i].predict(),'-')
    print model[i].params[15]

# Für Training alle degrees von 1 bis 15 testen:
polynomial_train = {}
polynomial_validate = {}
model_train = {}
RSS_train = {}
RSS_validate = {}
for i in range(1,16):
    # Bis auf i-te Polynom bilden:
    polynomial_train[i] = polynomial_dataframe(house_train_data[
            'sqft_living'],i)
    # Trainingsdatensatz Häuserpreise hinzuspielen:
    polynomial_train[i]['price'] = house_train_data['price']
    # Model bis i-ten Polynom bestimmen:
    # Problem(!!!): Nur bis viertes korrekt
    model_train[i] = sm.OLS(polynomial_train[i]['price'],
               sm.add_constant(polynomial_train[i].drop(['price'],
                               axis=1))).fit()
   
    # Werte vorhersagen:
    predict_i = 'predict_' + str(i)
    polynomial_train[i][predict_i] = model_train[i].predict()
    RSS_train[i] = (polynomial_train[i]['price'] - polynomial_train[i][predict_degree])**2
    RSS_train[i] = RSS_train[i].sum()
    
    # Predict Validation:
    polynomial_validate[i] = sm.add_constant(polynomial_dataframe(
            house_valid_data['sqft_living'],i))
    polynomial_validate[i]['price'] = house_valid_data['price']
    
    # Vorhersagen des Validation-Datasets:
    house_valid_data[predict_degree] = model_train[i].predict(
            polynomial_validate[i].drop(['price'],axis=1))
    # Error Validation:
    error_sq_degree = 'error_' + str(i)
    house_valid_data[error_sq_degree] = (house_valid_data['price'] - 
                    house_valid_data[predict_degree])**2
    RSS_validate[i] = house_valid_data[error_sq_degree].sum()
 
# Minimum RSS:
best_degree = min(RSS_validate,key=RSS_validate.get)

# RSS on Test Data with degree 4:
polynomial_test = sm.add_constant(polynomial_dataframe(house_test_data['sqft_living'],best_degree))
polynomial_test['price'] = house_test_data['price']
predict_degree = 'predict_' + str(best_degree)
house_test_data[predict_degree] = model_train[best_degree].predict(
        polynomial_test.drop(['price'],axis=1))
# Error Validation:
error_sq_degree = 'error_' + str(best_degree)
house_test_data[error_sq_degree] = (house_test_data['price'] - house_test_data[predict_degree])**2
RSS_test = house_test_data[error_sq_degree].sum()
    
from decimal import Decimal
'%.2E' % Decimal('138941857926817.86')    
    