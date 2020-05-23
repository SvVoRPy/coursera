import os
path = 'C:/Users/SvenV/Desktop/Coursera Kurse/Machine Learning Spezialization II/Assignments'
os.chdir(path)
os.listdir(path)

import pandas as pd

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

house_data = pd.read_csv('kc_house_data.csv',dtype=dtype_dict)
house_test_data = pd.read_csv('kc_house_test_data.csv',dtype=dtype_dict)
house_train_data = pd.read_csv('kc_house_train_data.csv',dtype=dtype_dict)

from scipy import stats

def simple_linear_regression(input_feature, output):
    temp_ols = stats.linregress(input_feature,output)
    return(temp_ols.intercept,temp_ols.slope)

sqfeet_intercept = simple_linear_regression(house_train_data['sqft_living'],house_train_data['price'])[0]
sqfeet_slope = simple_linear_regression(house_train_data['sqft_living'],house_train_data['price'])[1]
   
# House of 2650 sqft, prediction:
sqfeet_intercept+2650*sqfeet_slope

def get_residual_sum_of_squares(input_feature, output, intercept,slope):
    temp_predicted = intercept+slope*input_feature
    temp_sq_res = (output-temp_predicted)**2
    RSS=sum(temp_sq_res)
    return(RSS)

def inverse_regression_predictions(output, intercept, slope):
    estimated_input = (output- intercept)/slope
    return(estimated_input)

inverse_regression_predictions(800000,
                               sqfeet_intercept,
                               sqfeet_slope)

# -> 3004.3962451522771

bedrooms_intercept = simple_linear_regression(house_train_data['bedrooms'], house_train_data['price'])[0]
bedrooms_slope = simple_linear_regression(house_train_data['bedrooms'], house_train_data['price'])[1]

# RSS sqft_living:
RSS_sqft_living = get_residual_sum_of_squares(house_train_data['sqft_living'],
                            house_train_data['price'],
                            sqfeet_intercept,
                            sqfeet_slope)
# RSS bedrooms:
RSS_bedrooms = get_residual_sum_of_squares(house_train_data['bedrooms'],
                            house_train_data['price'],
                            bedrooms_intercept,
                            bedrooms_slope)

RSS_sqft_living<RSS_bedrooms