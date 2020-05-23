import os
path = 'C:/Users/SvenV/Desktop/Coursera Kurse/Machine Learning Spezialization II/Assignments'
os.chdir(path)
os.listdir(path)

import pandas as pd
import numpy as np

dtype_dict = dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int,
                           'sqft_living15':float, 'grade':int, 'yr_renovated':int,
                           'price':float, 'bedrooms':float, 'zipcode':str,
                           'long':float, 'sqft_lot15':float, 'sqft_living':float,
                           'floors':float, 'condition':int, 'lat':float, 'date':str,
                           'sqft_basement':int, 'yr_built':int, 'id':str,
                           'sqft_lot':int, 'view':int}

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

def predict_output(feature_matrix, weights):
    predictions = np.dot(feature_matrix,weights)
    return(predictions)

# Normalize the features
def normalize_features(features):
    norms = np.linalg.norm(features, axis=0)
    normalized_features = features / norms    
    return (normalized_features, norms)

(feature_matrix, output) = get_numpy_data(sales, ['sqft_living','bedrooms'], 'price')

(normalized_features, norms) = normalize_features(feature_matrix)
initial_weights = [1,4,1]

prediction = predict_output(normalized_features,initial_weights)

ro = []

for i in range(normalized_features.shape[1]):
    ro.append(sum(normalized_features[:,i]*(output - prediction
      + initial_weights[i]*normalized_features[:,i])))

range_lower =  80966698.666239053*2
range_upper = 87939470.823251516*2-0.001

# Both when l1_penalty größer/gleich 87939470.823251516*2


def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    #
    # compute prediction
    prediction = predict_output(feature_matrix,weights)
    # compute ro[i] = SUM[ [feature_i]*(output - prediction +
                 # weight[i]*[feature_i]) ]
    ro_i = sum(feature_matrix[:,i]*(output - prediction
      + weights[i]*feature_matrix[:,i]))
    
    if i == 0: # intercept -- do not regularize
        weights[i] = ro_i
    elif ro_i < -l1_penalty/2.:
        weights[i] = ro_i + l1_penalty/2
    elif ro_i > l1_penalty/2.:
        weights[i] = ro_i - l1_penalty/2
    else:
        weights[i] = 0.
    
    return weights[i]


def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, 
                                      l1_penalty, tolerance):
    weights = np.array(initial_weights,dtype=np.float)
    max_change = tolerance+1
    while max_change > tolerance:
        change = []
        for i in range(len(weights)):
            start_weight = weights[i]
            end_weight = lasso_coordinate_descent_step(i, feature_matrix, output,
                                                       weights, l1_penalty)
            change.append(abs(start_weight - end_weight))
        max_change = max(change)
        print max_change
    return weights

initial_weights = np.repeat(0,normalized_features.shape[1])
l1_penalty = 1e7
tolerance = 1.0

end_weights = lasso_cyclical_coordinate_descent(normalized_features, output, 
                                                initial_weights, l1_penalty, tolerance)

RSS = sum((output - predict_output(normalized_features,end_weights))**2)

#### 
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront'
            , 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built',
            'yr_renovated']
(feature_matrix, output) = get_numpy_data(train_data, features, 'price')
(normalized_features, norms) = normalize_features(feature_matrix)

l1_penalty=1e7
initial_weights = np.repeat(0,normalized_features.shape[1])
tolerance = 1

weights1e7 = lasso_cyclical_coordinate_descent(normalized_features, output,
                                               initial_weights, l1_penalty, tolerance)

dict_weights1e7 = dict(zip(['const'] + features, weights1e7.tolist()))

####
l1_penalty=1e8
weights1e8 = lasso_cyclical_coordinate_descent(normalized_features, output,
                                               initial_weights, l1_penalty, tolerance)

dict_weights1e8 = dict(zip(['const'] + features, weights1e8.tolist()))

###
l1_penalty=1e4
tolerance=5e5
weights1e4 = lasso_cyclical_coordinate_descent(normalized_features, output,
                                               initial_weights, l1_penalty, tolerance)

dict_weights1e4 = dict(zip(['const'] + features, weights1e4.tolist()))

# Rescale the weights to apply on test set
normalized_weights1e7 = weights1e7 / norms
normalized_weights1e8 = weights1e8 / norms
normalized_weights1e4 = weights1e4 / norms

(feature_matrix_test, output_test) = get_numpy_data(test_data, features, 'price')

RSS1e7 = RSS = sum((output_test - predict_output(feature_matrix_test,
                                                 normalized_weights1e7))**2)

RSS1e8 = RSS = sum((output_test - predict_output(feature_matrix_test,
                                                 normalized_weights1e8))**2)

RSS1e4 = RSS = sum((output_test - predict_output(feature_matrix_test,
                                                 normalized_weights1e4))**2)

min([RSS1e7, RSS1e8, RSS1e4])