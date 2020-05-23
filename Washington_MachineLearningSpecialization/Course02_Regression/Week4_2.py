import os
path = 'C:/Users/SvenV/Desktop/Coursera Kurse/Machine Learning Spezialization II/Assignments'
os.chdir(path)
os.listdir(path)

import pandas as pd
import numpy as np

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

sales = pd.read_csv('kc_house_data.csv',dtype=dtype_dict)
test_data = pd.read_csv('kc_house_test_data.csv',dtype=dtype_dict)
train_data = pd.read_csv('kc_house_train_data.csv',dtype=dtype_dict)


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
def predict_output(feature_matrix, weights):
    predictions = np.dot(feature_matrix,weights)
    return(predictions)

# Derivative regression cost function bilden:
def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant):
    derivative = None
    if feature_is_constant==True:
        derivative = 2*np.dot(errors, feature)
    else:        
        derivative = 2*np.dot(errors, feature) + 2*l2_penalty*weight
        
    return(derivative)  

# Gradient descent for Ridge Regression

def ridge_regression_gradient_descent(feature_matrix, output, initial_weights,
                                      step_size, l2_penalty, max_iterations=100):
    weights = np.array(initial_weights,dtype=np.float) # make sure it's a numpy array
    iterations = 0    
    while iterations < max_iterations:
        #while not reached maximum number of iterations:
        # compute the predictions using your predict_output() function
        predictions = predict_output(feature_matrix, weights)
        # compute the errors as predictions - output
        errors = np.array(predictions - output)
       
        for i in xrange(len(weights)): # loop over each weight
            # Recall that feature_matrix[:,i] is the feature column associated with weights[i]
            # compute the derivative for weight[i].
            #(Remember: when i=0, you are computing the derivative of the constant!)
            if i == 0:
                derivative = feature_derivative_ridge(errors, feature_matrix[:,i], weights[i], l2_penalty, True)
            else:
                derivative = feature_derivative_ridge(errors, feature_matrix[:,i], weights[i], l2_penalty, False)
            # subtract the step size times the derivative from the current weight  
            weights[i]  = weights[i] - (step_size * derivative)
            print 'Iteration: ' + str(iterations) + ' coef' + str(i) + '  ' + str(weights[i]) + ' derivative ' + str(derivative)
        
        iterations+=1
        
    return weights

simple_features = ['sqft_living']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
(simple_test_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)

# Define all for function
step_size = 1e-12
max_iterations = 1000
l2_penalty = 0
initial_weights = np.repeat(0,simple_feature_matrix.shape[1])

simple_weights_0_penalty = ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights, 
                                  step_size, l2_penalty, max_iterations)

l2_penalty = 1e11
simple_weights_high_penalty = ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights, 
                                  step_size, l2_penalty, max_iterations)

### Plot the two models
import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(simple_feature_matrix,output,'k.',
        simple_feature_matrix,predict_output(simple_feature_matrix, simple_weights_0_penalty),'b-',
        simple_feature_matrix,predict_output(simple_feature_matrix, simple_weights_high_penalty),'r-')


## Compute the RSS for the Test Data
# Initial Weights:
RSS_Initial = sum((test_output-predict_output(simple_test_feature_matrix,
                                              initial_weights))**2)
RSS_NoRegular = sum((test_output-predict_output(simple_test_feature_matrix,
                                                simple_weights_0_penalty))**2)
RSS_WithRegular = sum((test_output-predict_output(simple_test_feature_matrix,
                                                  simple_weights_high_penalty))**2)


##### Multiple Regression with two features
model_features = ['sqft_living', 'sqft_living15']
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
(test_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)

step_size = 1e-12
max_iterations = 1000
l2_penalty = 0
initial_weights = np.repeat(0,feature_matrix.shape[1])

multiple_weights_0_penalty = ridge_regression_gradient_descent(feature_matrix, output, initial_weights, 
                                  step_size, l2_penalty, max_iterations)

l2_penalty = 1e11
multiple_weights_high_penalty = ridge_regression_gradient_descent(feature_matrix, output, initial_weights, 
                                  step_size, l2_penalty, max_iterations)

### RSS for Test Data:
RSS_Multiple_Initial = sum((test_output-predict_output(test_feature_matrix,initial_weights))**2)
RSS_Multiple_NoRegular = sum((test_output-predict_output(test_feature_matrix,multiple_weights_0_penalty))**2)
RSS_Multiple_WithRegular = sum((test_output-predict_output(test_feature_matrix,multiple_weights_high_penalty))**2)

test_output[0]-predict_output(test_feature_matrix[0],multiple_weights_0_penalty)
test_output[0]-predict_output(test_feature_matrix[0],multiple_weights_high_penalty)