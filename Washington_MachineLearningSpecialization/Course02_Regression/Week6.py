import os
path = 'C:/Users/SvenV/Desktop/Coursera Kurse/Machine Learning Spezialization II/Assignments'
os.chdir(path)
os.listdir(path)

import pandas as pd
import numpy as np

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float,
              'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':int,
              'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int,
              'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str,
              'sqft_lot':int, 'view':int}

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

### Data
data = pd.read_csv('kc_house_data_small.csv',dtype=dtype_dict)
features = list(data.drop(['id','date','price'],axis=1))
output = 'price'

(features_train, output_train) = get_numpy_data(pd.read_csv('kc_house_data_small_train.csv',dtype=dtype_dict),
                                   features, output)
(features_validation, output_validation) = get_numpy_data(pd.read_csv('kc_house_data_small_validation.csv',dtype=dtype_dict),
                                   features, output)
(features_test, output_test) = get_numpy_data(pd.read_csv('kc_house_data_small_test.csv',dtype=dtype_dict),
                                  features, output)


# Normalize the features
def normalize_features(features):
    norms = np.linalg.norm(features, axis=0)
    normalized_features = features / norms    
    return (normalized_features, norms)

(features_train, norms) = normalize_features(features_train)
features_test = features_test / norms
features_validation = features_validation / norms

print features_test[0]
print features_train[9]

abs(features_test[0]-features_train[9]).sum()

# Euclidean Distance Query House to the first 10 others
distances = abs(features_test[0]-features_train[0:10]).sum(axis=1)
distances.argmin(axis=0)

for i in xrange(3):
    print features_train[i]-features_test[0]
    # should print 3 vectors of length 18

print features_train[0:3] - features_test[0]

# verify that vectorization works
results = features_train[0:3] - features_test[0]
print results[0] - (features_train[0]-features_test[0])
# should print all 0's if results[0] == (features_train[0]-features_test[0])
print results[1] - (features_train[1]-features_test[0])
# should print all 0's if results[1] == (features_train[1]-features_test[0])
print results[2] - (features_train[2]-features_test[0])
# should print all 0's if results[2] == (features_train[2]-features_test[0])


# Euclidean Distance Query House to all others
def compute_distances(features_instances, features_query):
    diff = features_instances-features_query
    distances = np.sqrt((diff**2).sum(axis=1))
    return distances

NN = compute_distances(features_train,features_test[2]).argmin(axis=0)
output_train[NN]

def k_nearest_neighbors(k, features_instances, features_query):
    diff = features_instances-features_query
    distances = list(np.sqrt((diff**2).sum(axis=1)))
    temp = list(distances)
    temp.sort()
    neighbors = map(distances.index,temp[0:k])
    return neighbors

k_nearest_neighbors(4, features_train, features_test[2])

def predict_output_of_query(k, features_train, output_train, features_query):
    neighbors = k_nearest_neighbors(k, features_train, features_query)
    prediction = output_train[neighbors].mean()
    return prediction

predict_output_of_query(4, features_train, output_train, features_test[2])

def predict_output(k, features_train, output_train, features_query):
    predictions = []
    for query in features_query:
        predictions.append(predict_output_of_query(k, features_train, output_train, query))
    return predictions

predictions = predict_output(10, features_train, output_train,
                         features_test[0:9])
position = np.argmin(predictions,axis=0)
position
predictions[position]


# Best k
RSS = []
for k in range(1,16):
    predictions = predict_output(k, features_train, 
                                 output_train, features_validation)
    RSS.append(((output_validation-predictions)**2).sum())

RSS.index(min(RSS))

### k=7

predictions_test = predict_output(7, features_train, output_train, features_test)

((output_test-predictions_test)**2).sum()
