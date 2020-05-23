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


def get_numpy_data(data_frame, simple_features, my_output):
    data_frame['constant'] = 1 # add a constant column to an SFrame
    # prepend variable 'constant' to the features list
    simple_features = ['constant'] + simple_features
    # select the columns of data_SFrame given by the ‘features’ list into the SFrame ‘features_sframe’

    # this will convert the features_sframe into a numpy matrix with GraphLab Create >= 1.7!!
    features_matrix = np.array(data_frame[simple_features])
    # assign the column of data_sframe associated with the target to the variable ‘output_sarray’

    # this will convert the SArray into a numpy array:
    output_array = np.array(data_frame[my_output]) # GraphLab Create>= 1.7!!
    return(features_matrix, output_array)
  
# Funktion definieren
def predict_outcome(features_matrix, weights):
    predictions = np.dot(features_matrix,weights)
    return(predictions)

# Derivative regression cost function bilden:
def feature_derivative(errors, feature):
    derivative = -2*np.dot(errors,feature)
    return(derivative)  

# Function for gradient descent
def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False
    weights = np.array(initial_weights)
    while not converged:
        predictions = predict_outcome(feature_matrix, weights)
        errors = np.array(output - predictions)
        
        gradient_sum_squares = 0 # initialize the gradient
        # while not converged, update each weight individually:
        for i in range(len(weights)):
            # Recall that feature_matrix[:, i] is the feature column associated with weights[i]
            # compute the derivative for weight[i]:
            # add the squared derivative to the gradient magnitude
            temp_derivative = feature_derivative(errors,feature_matrix[:, i])
            gradient_sum_squares += temp_derivative**2
            # update the weight based on step size and derivative:
            weights[i] = weights[i] - (step_size*temp_derivative)
            
        gradient_magnitude = np.sqrt(gradient_sum_squares)
        print gradient_magnitude
        if gradient_magnitude < tolerance:
            converged = True
    return(weights)

simple_features = ['sqft_living']
my_output= 'price'
initial_weights = np.array([-47000., 1.])
step_size = 7e-12
tolerance = 2.5e7

(simple_feature_matrix, output) = get_numpy_data(house_train_data, simple_features, my_output)

simple_weights = regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, tolerance)

# Kontrolle über originäre Funktion:
#import statsmodels.api as sm

#sm.OLS(house_train_data['price'],sm.add_constant(house_train_data[['sqft_living']])).fit().params

(test_simple_feature_matrix, test_output) = get_numpy_data(house_test_data, simple_features, my_output)

# Predictes Price for 1st House
predictions_test_simple = predict_outcome(test_simple_feature_matrix, simple_weights)
predictions_test_simple[0].round()

# Residual Sum of Squares:
RSS_test_simple = (np.array(house_test_data['price'] - predictions_test_simple)**2).sum()

#### New multiple model
model_features = ['sqft_living','sqft_living15']
my_output = 'price'
initial_weights = np.array([-100000., 1., 1.])
step_size = 4e-12
tolerance = 1e9

(model_feature_matrix, output) = get_numpy_data(house_train_data, model_features, my_output)

model_weights = regression_gradient_descent(model_feature_matrix, output, initial_weights, step_size, tolerance)

# Prediction for the 1st house:
(test_model_feature_matrix, test_output) = get_numpy_data(house_test_data, model_features, my_output)
predictions_test_model = predict_outcome(test_model_feature_matrix, model_weights)
predictions_test_model[0].round()    
# Actual Price
house_test_data['price'][0]
# Model 1 error:
house_test_data['price'][0]-predictions_test_simple[0].round()
# Model 2 error:
house_test_data['price'][0]-predictions_test_model[0].round() 

# RSS test data for model 2:
RSS_test_model = (np.array(house_test_data['price'] - predictions_test_model)**2).sum()

# Lower RSS for model 2?
RSS_test_simple>RSS_test_model

