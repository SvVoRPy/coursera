def intermediate_node_num_mistakes(labels_in_node):
    # Corner case: If labels_in_node is empty, return 0
    if len(labels_in_node) == 0:
        return 0    
    # Count the number of 1's (safe loans)
    safe_loans = sum(labels_in_node == 1) 
    # Count the number of -1's (risky loans)
    risky_loans = sum(labels_in_node == -1) 
    # Return the number of mistakes that the majority classifier makes.
    return len(labels_in_node) - max(safe_loans,risky_loans)

def best_splitting_feature(data, features, target):
    
    target_values = data[target]
    best_feature = None # Keep track of the best feature 
    best_error = 10     # Keep track of the best error so far 
    # Note: Since error is always <= 1, we should intialize it with something larger than 1.

    # Convert to float to make sure error gets computed correctly.
    num_data_points = float(len(data))  
    
    # Loop through each feature to consider splitting on that feature
    for feature in features:
        
        # The left split will have all data points where the feature value is 0
        left_split = data[data[feature] == 0]
        
        # The right split will have all data points where the feature value is 1
        right_split =  data[data[feature] == 1]
            
        # Calculate the number of misclassified examples in the left split.
        left_mistakes = intermediate_node_num_mistakes(left_split[target])            

        # Calculate the number of misclassified examples in the right split.
        right_mistakes = intermediate_node_num_mistakes(right_split[target]) 
            
        # Compute the classification error of this split.
        # Error = (# of mistakes (left) + # of mistakes (right)) / (# of data points)
        error = (left_mistakes + right_mistakes)/num_data_points

        # If this is the best error we have found so far, store the feature as best_feature and the error as best_error
        if error < best_error:
            best_error = error
            best_feature = feature        
    
    return best_feature # Return the best feature we found


def create_leaf(target_values):    
    # Create a leaf node
    leaf = {'splitting_feature' : None,
            'left' : None,
            'right' : None,
            'is_leaf': True}  
   
    # Count the number of data points that are +1 and -1 in this node.
    num_ones = len(target_values[target_values == +1])
    num_minus_ones = len(target_values[target_values == -1])    

    # For the leaf node, set the prediction to be the majority class.
    # Store the predicted class (1 or -1) in leaf['prediction']
    if num_ones > num_minus_ones:
        leaf['prediction'] = +1
    else:
        leaf['prediction'] = -1

    # Return the leaf node
    return leaf 

def reached_minimum_node_size(data, min_node_size):
    # Return True if the number of data points is less than or equal to the minimum node size.
    if len(data) <= min_node_size:
        return True
    
def error_reduction(error_before_split, error_after_split):
    # Return the error before the split minus the error after the split.
    return error_before_split - error_after_split


def decision_tree_create(data, features, target, current_depth, max_depth, min_node_size, min_error_reduction):
    remaining_features = features[:] # Make a copy of the features.
    
    target_values = data[target]
    print "--------------------------------------------------------------------"
    print "Subtree, depth = %s (%s data points)." % (current_depth, len(target_values))
    

    # Stopping condition 1
    # (Check if there are mistakes at current node.
    # Recall you wrote a function intermediate_node_num_mistakes to compute this.)
    if intermediate_node_num_mistakes(target_values) == 0:
        print "Stopping condition 1 reached."     
        # If not mistakes at current node, make current node a leaf node
        return create_leaf(target_values)
    
    # Stopping condition 2 (check if there are remaining features to consider splitting on)
    if remaining_features == 0:
        print "Stopping condition 2 reached."    
        # If there are no remaining features to consider, make current node a leaf node
        return create_leaf(target_values)    
    
    # Additional stopping condition (limit tree depth)
    if current_depth >= max_depth: 
        print "Reached maximum depth. Stopping for now."
        # If the max tree depth has been reached, make current node a leaf node
        return create_leaf(target_values)
    
    if reached_minimum_node_size(target_values, min_node_size) == True:
        print "Early stopping condition 2 reached. Reached minimum node size."
        return create_leaf(target_values)

    # Find the best splitting feature (recall the function best_splitting_feature implemented above)
    splitting_feature = best_splitting_feature(data, features, target)
    
    # Split on the best feature that we found. 
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]
    
    # Early stopping condition 3: Minimum error reduction
    # Calculate the error before splitting (number of misclassified examples 
    # divided by the total number of examples)
    error_before_split = intermediate_node_num_mistakes(target_values) / float(len(data))
    
    # Calculate the error after splitting (number of misclassified examples 
    # in both groups divided by the total number of examples)
    left_mistakes =  intermediate_node_num_mistakes(left_split[target]) 
    right_mistakes =  intermediate_node_num_mistakes(right_split[target]) 
    error_after_split = (left_mistakes + right_mistakes) / float(len(data))
    
    # If the error reduction is LESS THAN OR EQUAL TO min_error_reduction, return a leaf.
    if  error_reduction(error_before_split, error_after_split) <= min_error_reduction:
        print "Early stopping condition 3 reached. Minimum error reduction."
        return create_leaf(target_values)
    
    
    remaining_features.remove(splitting_feature)
    print "Split on feature %s. (%s, %s)" % (\
                      splitting_feature, len(left_split), len(right_split))
    
    # Create a leaf node if the split is "perfect"
    if len(left_split) == len(data):
        print "Creating leaf node."
        return create_leaf(left_split[target])
    if len(right_split) == len(data):
        print "Creating leaf node."
        return create_leaf(right_split[target])

        
    # Repeat (recurse) on left and right subtrees
    left_tree = decision_tree_create(left_split, remaining_features, target, current_depth + 1, max_depth, min_node_size, min_error_reduction)        
    right_tree = decision_tree_create(right_split, remaining_features, target, current_depth + 1, max_depth, min_node_size, min_error_reduction)        

    return {'is_leaf'          : False, 
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree, 
            'right'            : right_tree}

def classify(tree, x, annotate = False):
       # if the node is a leaf node.
    if tree['is_leaf']:
        if annotate:
             print "At leaf, predicting %s" % tree['prediction']
        return tree['prediction']
    else:
        # split on feature.
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
    

def count_leaves(tree):
    if tree['is_leaf']:
        return 1
    return count_leaves(tree['left']) + count_leaves(tree['right'])