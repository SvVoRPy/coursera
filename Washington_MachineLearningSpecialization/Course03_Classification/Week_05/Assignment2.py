import os
path = 'C:/Users/SvenV/Desktop/Coursera Kurse/Machine Learning Spezialization III/Assignments/Week 5'
os.chdir(path)
os.listdir(path)

import pandas as pd
import numpy as np

loans = pd.read_csv('lending-club-data.csv')

loans.columns
loans.head(2)

loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.drop('bad_loans',axis=1)

features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
           ]
target = 'safe_loans'
loans = loans[features + [target]]

# Encode data for model
dummies = pd.get_dummies(loans.select_dtypes(include=[object]))

all_data = loans.select_dtypes(exclude=[object]).join(dummies)

# Split into Training and Validation
train_data = all_data.iloc[pd.read_json('module-8-assignment-2-train-idx.json')[0],:]
test_data = all_data.iloc[pd.read_json('module-8-assignment-2-test-idx.json')[0],:]

def intermediate_node_weighted_mistakes(labels_in_node, data_weights):
    # Sum the weights of all entries with label +1
    total_weight_positive = sum(data_weights[labels_in_node == +1])
    
    # Weight of mistakes for predicting all -1's is equal to the sum above
    weighted_mistakes_all_negative = total_weight_positive
    
    # Sum the weights of all entries with label -1
    total_weight_negative = sum(data_weights[labels_in_node == -1])
    
    # Weight of mistakes for predicting all +1's is equal to the sum above
    weighted_mistakes_all_positive = total_weight_negative
    
    # Return the tuple (weight, class_label) representing the lower of the two weights
    #    class_label should be an integer of value +1 or -1.
    # If the two weights are identical, return (weighted_mistakes_all_positive,+1)
    if weighted_mistakes_all_negative > weighted_mistakes_all_positive:
        return (weighted_mistakes_all_positive, 1)
    elif weighted_mistakes_all_negative < weighted_mistakes_all_positive:
        return (weighted_mistakes_all_negative, -1)
    else:
        return (weighted_mistakes_all_positive, 1)
    
    
        
def best_splitting_feature(data, features, target, data_weights):
    
    best_feature = None # Keep track of the best feature 
    best_error = 10     # Keep track of the best error so far 
    num_points = float(len(data))
    # Note: Since error is always <= 1, we should intialize it with something larger than 1.

    # Loop through each feature to consider splitting on that feature
    for feature in features:
        
        # The left split will have all data points where the feature value is 0
        left_split = data[data[feature] == 0]
        # The right split will have all data points where the feature value is 1
        right_split =  data[data[feature] == 1]
        
        # Same for the data_weights
        left_data_weights = data_weights[data[feature] == 0]
        right_data_weights = data_weights[data[feature] == 1]
            
        # Calculate the number of misclassified examples in the left split.
        left_weighted_mistakes, left_class = intermediate_node_weighted_mistakes(
                left_split[target], left_data_weights)
        right_weighted_mistakes, right_class = intermediate_node_weighted_mistakes(
                right_split[target], right_data_weights)
            
        # Compute the classification error of this split.
        # Error = (# of mistakes (left) + # of mistakes (right)) / (# of data points)
        error = (left_weighted_mistakes + right_weighted_mistakes)/num_points

        # If this is the best error we have found so far, store the feature as best_feature and the error as best_error
        if error < best_error:
            best_error = error
            best_feature = feature        
    
    return best_feature # Return the best feature we found  
    
    
def create_leaf(target_values, data_weights):
    
    # Create a leaf node
    leaf = {'splitting_feature' : None,
            'is_leaf': True}
    
    # Computed weight of mistakes.
    # Store the predicted class (1 or -1) in leaf['prediction']
    weighted_error, best_class = intermediate_node_weighted_mistakes(target_values, data_weights)
    leaf['prediction'] = best_class
    
    return leaf    
    
def weighted_decision_tree_create(data, features, target, data_weights, current_depth = 1, max_depth = 10):
    remaining_features = features[:] # Make a copy of the features.
    target_values = data[target]
    print "--------------------------------------------------------------------"
    print "Subtree, depth = %s (%s data points)." % (current_depth, len(target_values))
    
    # Stopping condition 1. Error is 0.
    if intermediate_node_weighted_mistakes(target_values, data_weights)[0] <= 1e-15:
        print "Stopping condition 1 reached."                
        return create_leaf(target_values, data_weights)
    
    # Stopping condition 2. No more features.
    if remaining_features == []:
        print "Stopping condition 2 reached."                
        return create_leaf(target_values, data_weights)    
    
    # Additional stopping condition (limit tree depth)
    if current_depth > max_depth:
        print "Reached maximum depth. Stopping for now."
        return create_leaf(target_values, data_weights)
    
    # If all the datapoints are the same, splitting_feature will be None. Create a leaf
    splitting_feature = best_splitting_feature(data, features, target, data_weights)
    remaining_features.remove(splitting_feature)
        
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]
    
    left_data_weights = data_weights[data[splitting_feature] == 0]
    right_data_weights = data_weights[data[splitting_feature] == 1]
    
    print "Split on feature %s. (%s, %s)" % (\
              splitting_feature, len(left_split), len(right_split))
    
    # Create a leaf node if the split is "perfect"
    if len(left_split) == len(data):
        print "Creating leaf node."
        return create_leaf(left_split[target], data_weights)
    if len(right_split) == len(data):
        print "Creating leaf node."
        return create_leaf(right_split[target], data_weights)
    
    # Repeat (recurse) on left and right subtrees
    left_tree = weighted_decision_tree_create(
        left_split, remaining_features, target, left_data_weights, current_depth + 1, max_depth)
    right_tree = weighted_decision_tree_create(
        right_split, remaining_features, target, right_data_weights, current_depth + 1, max_depth)
    
    return {'is_leaf'          : False, 
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree, 
            'right'            : right_tree}    
    
    
def count_nodes(tree):
    if tree['is_leaf']:
        return 1
    return 1 + count_nodes(tree['left']) + count_nodes(tree['right'])

def classify(tree, x, annotate = False):   
    # If the node is a leaf node.
    if tree['is_leaf']:
        if annotate: 
            print "At leaf, predicting %s" % tree['prediction']
        return tree['prediction'] 
    else:
        # Split on feature.
        split_feature_value = x[tree['splitting_feature']]
        if annotate: 
            print "Split on %s = %s" % (tree['splitting_feature'], split_feature_value)
        if split_feature_value == 0:
            return classify(tree['left'], x, annotate)
        else:
            return classify(tree['right'], x, annotate)
        
def evaluate_classification_error(tree, data):
    # Apply the classify(tree, x) to each row in your data
    predictions = list()
    for i in range(len(data)):
        predictions.append(classify(tree,data.iloc[i],annotate=False))
    # Once you've made the predictions, calculate the classification error and return it
    predictions = pd.DataFrame({'True': list(data['safe_loans']),'Pred': predictions})
    return sum(predictions['True'] == predictions['Pred'])/float(len(predictions))

# Assign weights
example_data_weights = np.array([1.] * 10 + [0.]*(len(train_data) - 20) + [1.] * 10)
# Train a weighted decision tree model.
features = list(train_data.columns)[1:]
small_data_decision_tree_subset_20 = weighted_decision_tree_create(train_data, features, target,
                         example_data_weights, max_depth=2)

evaluate_classification_error(small_data_decision_tree_subset_20, train_data)
evaluate_classification_error(small_data_decision_tree_subset_20, train_data[0:3])

#### Implementing own Adaboost Function

from math import log
from math import exp

def adaboost_with_tree_stumps(data, features, target, num_tree_stumps):
    # start with unweighted data
    alpha = np.array([1.]*len(data))
    weights = []
    tree_stumps = []
    target_values = data[target]
    
    for t in xrange(num_tree_stumps):
        print '====================================================='
        print 'Adaboost Iteration %d' % t
        print '====================================================='        
        # Learn a weighted decision tree stump. Use max_depth=1
        tree_stump = weighted_decision_tree_create(data, features, target, data_weights=alpha, max_depth=1)
        tree_stumps.append(tree_stump)
        
        # Make predictions
        predictions = list()
        for i in range(len(data)):
            predictions.append(classify(tree,data.iloc[i],annotate=False))
        
        # Produce a Boolean array indicating whether
        # each data point was correctly classified
        is_correct = predictions == target_values
        is_wrong   = predictions != target_values
        
        # Compute weighted error
        weighted_error = sum(is_wrong)/sum(alpha)
        
        # Compute model coefficient using weighted error
        weight = 0.5*log((1-weighted_error)/weighted_error)
        weights.append(weight)
        
        # Adjust weights on data point
        adjustment = is_correct.apply(lambda is_correct : exp(-weight) if is_correct else exp(weight))
        
        # Scale alpha by multiplying by adjustment
        # Then normalize data points weights
        alpha = alpha*adjustment
    
    return weights, tree_stumps

features = list(train_data.columns)[1:]
adaboost = adaboost_with_tree_stumps(train_data,features,target,num_tree_stumps=10)

def predict_adaboost(stump_weights, tree_stumps, data):
    scores = pd.Series([0.]*len(data))
    
    for i, tree_stump in enumerate(tree_stumps):
        predictions = list()
        for n in range(len(data)):
            predictions.append(classify(tree_stump,data.iloc[n],annotate=False))
        
        # Accumulate predictions on scaores array
        scores += pd.Series(stump_weights[0]*np.array(predictions))
        
    return scores.apply(lambda score : +1 if score > 0 else -1)

adaboost2 = adaboost_with_tree_stumps(train_data,features,target,num_tree_stumps=30)

from sklearn.metrics import accuracy_score
error_all = []
for n in xrange(1, 31):
    predictions = predict_adaboost(stump_weights[:n], tree_stumps[:n], train_data)
    error = 1.0 - accuracy_score(train_data[target],predictions)
    error_all.append(error)
    print "Iteration %s, training error = %s" % (n, error_all[n-1])
