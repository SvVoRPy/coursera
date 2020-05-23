import os
path = 'C:/Users/SvenV/Desktop/Coursera Kurse/Machine Learning Spezialization II/Assignments'
os.chdir(path)
os.listdir(path)

import pandas as pd
import numpy as np

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

house_data = pd.read_csv('kc_house_data.csv',dtype=dtype_dict)
house_test_data = pd.read_csv('kc_house_test_data.csv',dtype=dtype_dict)
house_train_data = pd.read_csv('kc_house_train_data.csv',dtype=dtype_dict)

# New Variables:
house_train_data['bedrooms_squared'] = house_train_data['bedrooms']*house_train_data['bedrooms']
house_train_data['bed_bath_rooms'] = house_train_data['bedrooms']*house_train_data['bathrooms']
house_train_data['log_sqft_living'] = np.log(house_train_data['sqft_living'])
house_train_data['lat_plus_long'] = house_train_data['lat'] + house_train_data['long']

house_test_data['bedrooms_squared'] = house_test_data['bedrooms']*house_test_data['bedrooms']
house_test_data['bed_bath_rooms'] = house_test_data['bedrooms']*house_test_data['bathrooms']
house_test_data['log_sqft_living'] = np.log(house_test_data['sqft_living'])
house_test_data['lat_plus_long'] = house_test_data['lat'] + house_test_data['long']

# Mean values in Test Data for new Variables
house_test_data[['bedrooms_squared','bed_bath_rooms','log_sqft_living','lat_plus_long']].mean()

# Models on training data
import statsmodels.api as sm

m1 = sm.OLS(house_train_data['price'],sm.add_constant(house_train_data[['sqft_living',
       'bedrooms','bathrooms','lat','long']])).fit()
    
m2 = sm.OLS(house_train_data['price'],sm.add_constant(house_train_data[['sqft_living',
       'bedrooms','bathrooms','lat','long','bed_bath_rooms']])).fit()
    
m3 = sm.OLS(house_train_data['price'],sm.add_constant(house_train_data[['sqft_living',
       'bedrooms','bathrooms','lat','long','bed_bath_rooms',
       'bedrooms_squared','log_sqft_living','lat_plus_long']])).fit()

m1.params
m2.params

# Compute the RSS
# m1 training:
RSS_m1_training = ((house_train_data['price']-m1.predict(sm.add_constant(house_train_data[['sqft_living',
                'bedrooms','bathrooms','lat','long']])))**2).sum()
# m2 training:
RSS_m2_training = ((house_train_data['price']-m2.predict(sm.add_constant(house_train_data[['sqft_living',
       'bedrooms','bathrooms','lat','long','bed_bath_rooms']])))**2).sum()
# m3 training
RSS_m3_training = ((house_train_data['price']-m3.predict(sm.add_constant(house_train_data[['sqft_living',
       'bedrooms','bathrooms','lat','long','bed_bath_rooms',
       'bedrooms_squared','log_sqft_living','lat_plus_long']])))**2).sum()
    
# Test Datensatz:
RSS_m1_test = ((house_test_data['price']-m1.predict(sm.add_constant(house_test_data[['sqft_living',
                'bedrooms','bathrooms','lat','long']])))**2).sum()
# m2 training:
RSS_m2_test = ((house_test_data['price']-m2.predict(sm.add_constant(house_test_data[['sqft_living',
       'bedrooms','bathrooms','lat','long','bed_bath_rooms']])))**2).sum()
# m3 training
RSS_m3_test = ((house_test_data['price']-m3.predict(sm.add_constant(house_test_data[['sqft_living',
       'bedrooms','bathrooms','lat','long','bed_bath_rooms',
       'bedrooms_squared','log_sqft_living','lat_plus_long']])))**2).sum()

RSS_m1_training
RSS_m2_training
RSS_m3_training

RSS_m1_test
RSS_m2_test
RSS_m3_test

