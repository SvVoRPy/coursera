import os
path = 'C:/Users/SvenV/Desktop/Coursera Kurse/Machine Learning Spezialization II/Assignments'
os.chdir(path)
os.listdir(path)

import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
%matplotlib inline

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

sales = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)
sales = sales.sort(['sqft_living','price'])

# Polynomial Data-Frame
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

poly15_data = polynomial_dataframe(sales['sqft_living'],15)

# Penalty:
l2_small_penalty = 1.5e-5

model = linear_model.Ridge(alpha=l2_small_penalty, normalize=True)
feature_1 = model.fit(poly15_data, sales['price']).coef_[0]


# Fit 15th Polynomial Degree Model:
l2_small_penalty=1e-9
model_15 = linear_model.Ridge(alpha=l2_small_penalty, normalize=True)

# all four subsets:
set_1 = pd.read_csv('wk3_kc_house_set_1_data.csv', dtype=dtype_dict)
set_2 = pd.read_csv('wk3_kc_house_set_2_data.csv', dtype=dtype_dict)
set_3 = pd.read_csv('wk3_kc_house_set_3_data.csv', dtype=dtype_dict)
set_4 = pd.read_csv('wk3_kc_house_set_4_data.csv', dtype=dtype_dict)

house_set = {1: set_1 , 2: set_2 , 3: set_3 , 4: set_4}
daten = {}
models_sets = {}
predicts_sets = {}
weights_smallL2 = {}
for i in range(1,5):
    daten[i] = polynomial_dataframe(house_set[i]['sqft_living'],15)
    models_sets[i] = model_15.fit(daten[i], house_set[i]['price'])
    predicts_sets[i] = model_15.predict(daten[i])
    weights_smallL2[i] = pd.Series(models_sets[i].coef_)

# Grafische Darstellung
plt.subplots(2,2)
for i in range(1,5):
    ax=plt.subplot(2,2,i)
    ax.plot(daten[i]['power_1'], house_set[i]['price'],'.',
         daten[i]['power_1'], predicts_sets[i],'-')

# Weights:
weights_smallL2 = pd.DataFrame(weights).set_index([range(1,16)])
weights_smallL2.columns = ['House Set 1','House Set 2','House Set 3','House Set 4']

weights_smallL2.iloc[0]

###### Large L2 Penalty
l2_large_penalty=1.23e2
model_15_large = linear_model.Ridge(alpha=l2_large_penalty, normalize=True)
weights_largeL2 = {}
for i in range(1,5):
    models_sets[i] = model_15_large.fit(daten[i], house_set[i]['price'])
    predicts_sets[i] = model_15_large.predict(daten[i])
    weights_largeL2[i] = pd.Series(models_sets[i].coef_)

plt.subplots(2,2)
for i in range(1,5):
    ax=plt.subplot(2,2,i)
    ax.plot(daten[i]['power_1'], house_set[i]['price'],'.',
         daten[i]['power_1'], predicts_sets[i],'-')

# Weights:
weights_largeL2 = pd.DataFrame(weights_largeL2).set_index([range(1,16)])
weights_largeL2.columns = ['House Set 1','House Set 2','House Set 3','House Set 4']

weights_largeL2.iloc[0]

##### Selecting an L2 penalty via cross-validation

train_valid_shuffled = pd.read_csv('wk3_kc_house_train_valid_shuffled.csv', dtype=dtype_dict)
test = pd.read_csv('wk3_kc_house_test_data.csv', dtype=dtype_dict)

# Divide training set into k equal sized segments
# Each Segment: n/k observations
# segment 0: starts at index 0 and contains n/k elements, it ends at index (n/k)-1

n = len(train_valid_shuffled)
k = 10 # 10-fold cross-validation

for i in xrange(k):
    start = (n*i)/k
    end = (n*(i+1))/k-1
    print i, (start, end)
    
def k_fold_cross_validation(k, l2_penalty, data, output):
    validation_error = []
    for i in xrange(k):
        start = (n*i)/k
        end = (n*(i+1))/k
        validation_data = data[start:end+1]
        validation_output = output[start:end+1]
        train_data = data[0:start].append(data[end+1:n])
        train_output = output[0:start].append(output[end+1:n])
        # Fit the model with training data
        model = linear_model.Ridge(alpha=l2_penalty, normalize=True).fit(
                train_data, train_output)
        # Error based on validation:
        error_sq = (validation_output - model.predict(validation_data))**2
        validation_error.append(error_sq.sum())
    return sum(validation_error)/k

# Polynomial Data Generation:
poly15_train = polynomial_dataframe(train_valid_shuffled['sqft_living'],15)
# Which penalty is the best?
penalties = np.logspace(3, 9, num=13)
testing_penalties = {}
for i in penalties:
    testing_penalties[i] = k_fold_cross_validation(10, i, poly15_train, train_valid_shuffled['price'])

# Minimal RSS Penalty:    
l2_penalty = min(testing_penalties, key=testing_penalties.get)    

model = linear_model.Ridge(alpha=l2_penalty, normalize=True).fit(
        poly15_train, train_valid_shuffled['price'])

# RSS on Test Data:
test_poly15 = polynomial_dataframe(test['sqft_living'],15)

error_sq_test = (test['price'] - model.predict(test_poly15))**2
error_sq_test.sum()