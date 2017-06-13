# coding=utf-8
__author__ = 'nadyaK'
__date__ = '05/12/2017'

import ml_numpy_utils as np_utils
import graphlab as gp
from math import log, exp, sqrt

class LogisticRegression(object):
	"""simple linear regression model """

	def __init__(self):
		self.__name__ = "logistic_regression_model"

	def _check_iteration(self, itr):
		check_itr = False
		if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) or (
						itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
			check_itr = True
		return check_itr

	def _check_log_likelihood(self, itr, feature_matrix, sentiment, coefficients, max_iter):
		if self._check_iteration(itr):
			lp = np_utils.compute_log_likelihood(feature_matrix,sentiment, coefficients)
			print 'iteration %*d: log likelihood of observed labels = %.8f' % (
				int(np_utils.np.ceil(np_utils.np.log10(max_iter))),itr,lp)

	def logistic_regression(self, feature_matrix, sentiment, initial_coefficients,
							step_size, max_iter, check_likelihood=True):
		coefficients = np_utils.np.array(initial_coefficients) # make sure it's a numpy array
		for itr in xrange(max_iter):

			# Predict P(y_i = +1|x_i,w)
			predictions = np_utils.predict_probability(feature_matrix,coefficients)

			# Compute indicator value for (y_i = +1)
			indicator = (sentiment == +1) #otherwise false

			# Compute the errors as indicator - predictions
			errors = indicator - predictions

			for j in xrange(len(coefficients)): # loop over each coefficient
				#feature_matrix[:,j] is column associated with coefficients[j].
				# Compute the derivative for coefficients[j].
				feature = feature_matrix[:,j]
				derivative = np_utils.feature_derivative(errors,feature)

				# add the step size times the derivative to the current coefficient
				coefficients[j] += step_size * derivative

			if check_likelihood:
				self._check_log_likelihood(itr, feature_matrix, sentiment, coefficients, max_iter)

		return coefficients

	def _check_log_likelihood_with_L2(self, itr, feature_matrix, sentiment, coefficients, max_iter, l2_penalty):
		if self._check_iteration(itr):
			lp = np_utils.compute_log_likelihood_with_L2(feature_matrix,sentiment, coefficients, l2_penalty)
			print 'iteration %*d: log likelihood of observed labels = %.8f' % (
				int(np_utils.np.ceil(np_utils.np.log10(max_iter))),itr,lp)

	def logistic_regression_with_L2(self, feature_matrix, sentiment, initial_coefficients,
									step_size, l2_penalty, max_iter, check_likelihood=True):
		coefficients = np_utils.np.array(initial_coefficients) # make sure it's a numpy array

		for itr in xrange(max_iter):
			# Predict P(y_i = +1|x_i,w) using your predict_probability() function
			predictions = np_utils.predict_probability(feature_matrix,coefficients)

			# Compute indicator value for (y_i = +1)
			indicator = (sentiment == +1)

			# Compute the errors as indicator - predictions
			errors = indicator - predictions
			for j in xrange(len(coefficients)): # loop over each coefficient
				#is_intercept = (j == 0)
				is_constant = True if j == 0 else False
				# Recall that feature_matrix[:,j] is the feature column associated with coefficients[j].
				# Compute the derivative for coefficients[j]. Save it in a variable called derivative
				feature = feature_matrix[:,j]
				derivative = np_utils.feature_derivative_with_L2(errors,feature,coefficients[j],l2_penalty,is_constant)
				# add the step size times the derivative to the current coefficient
				coefficients[j] += step_size * derivative

			if check_likelihood:
				self._check_log_likelihood_with_L2(itr, feature_matrix, sentiment, coefficients, max_iter, l2_penalty)

		return coefficients

class DecisionTree(object):
	"""Decision Tree model """

	def __init__(self):
		self.__name__ = "decision_tree_model"

	def intermediate_node_num_mistakes(self, labels_in_node):
		# Corner case: If labels_in_node is empty, return 0
		if len(labels_in_node) == 0:
			return 0

		# Count the number of 1's (safe loans)
		positives = len(labels_in_node[labels_in_node == 1])   #(filter(lambda x: x == 1)).sum()

		# Count the number of -1's (risky loans)
		negatives = len(labels_in_node[labels_in_node == -1])

		# Return the number of mistakes that the majority classifier makes.
		majority = negatives if positives > negatives else positives

		return majority

	def best_splitting_feature(self, data, features, target):
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
			right_split = data[data[feature] == 1]

			# Calculate the number of misclassified examples in the left split.
			# Remember that we implemented a function for this! (It was called intermediate_node_num_mistakes)
			left_mistakes = self.intermediate_node_num_mistakes(left_split[target]) #target has all 1|-1  ....

			# Calculate the number of misclassified examples in the right split.
			right_mistakes = self.intermediate_node_num_mistakes(right_split[target])

			# Compute the classification error of this split.
			# Error = (# of mistakes (left) + # of mistakes (right)) / (# of data points)
			error = (left_mistakes + right_mistakes) / num_data_points

			# If this is the best error we have found so far, store the feature as best_feature and the error as best_error
			if error < best_error:
				best_feature = feature
				best_error = error

		return best_feature # Return the best feature we found

	def create_leaf(self, target_values):
		# Create a leaf node
		leaf = {'splitting_feature':None,'left':None,'right':None,'is_leaf':True}

		# Count the number of data points that are +1 and -1 in this node.
		num_ones = len(target_values[target_values == +1])
		num_minus_ones = len(target_values[target_values == -1])

		# For the leaf node, set the prediction to be the majority class.
		# Store the predicted class (1 or -1) in leaf['prediction']
		leaf['prediction'] = 1 if num_ones > num_minus_ones else -1

		# Return the leaf node
		return leaf

	def decision_tree_create(self, data, features, target, current_depth = 0,
							 max_depth = 10, min_node_size=1,
							 min_error_reduction=0.0, verbose=True):

		remaining_features = features[:] # Make a copy of the features.

		target_values = data[target]
		if verbose:
			print "--------------------------------------------------------------------"
			print "Subtree, depth = %s (%s data points)." % (current_depth,len(target_values))

		# Stopping condition 1: All nodes are of the same type.
		if self.intermediate_node_num_mistakes(target_values) == 0:
			if verbose: print "Stopping condition 1 reached. All data points have the same target value."
			return self.create_leaf(target_values)

		# Stopping condition 2: No more features to split on.
		if remaining_features == []:
			if verbose: print "Stopping condition 2 reached. No remaining features."
			return self.create_leaf(target_values)

		# Early stopping condition 1: Reached max depth limit.
		if current_depth >= max_depth:
			if verbose: print "Early stopping condition 1 reached. Reached maximum depth."
			return self.create_leaf(target_values)

		# Early stopping condition 2: Reached the minimum node size.
		# If the number of data points is less than or equal to the minimum size, return a leaf.
		if self.reached_minimum_node_size(target_values,min_node_size):
			if verbose: print "Early stopping condition 2 reached. Reached minimum node size."
			return self.create_leaf(target_values)

		# Find the best splitting feature
		splitting_feature = self.best_splitting_feature(data,features,target)

		# Split on the best feature that we found.
		left_split = data[data[splitting_feature] == 0]
		right_split = data[data[splitting_feature] == 1]

		# Early stopping condition 3: Minimum error reduction
		# Calculate the error before splitting (number of misclassified examples
		# divided by the total number of examples)
		error_before_split = self.intermediate_node_num_mistakes(target_values) / float(len(data))

		# Calculate the error after splitting (number of misclassified examples
		# in both groups divided by the total number of examples)
		left_mistakes = self.intermediate_node_num_mistakes(left_split[target])
		right_mistakes = self.intermediate_node_num_mistakes(right_split[target])
		error_after_split = (left_mistakes + right_mistakes) / float(len(data))

		# If the error reduction is LESS THAN OR EQUAL TO min_error_reduction, return a leaf.
		if self.error_reduction(error_before_split,error_after_split) <= min_error_reduction:
			if verbose: print "Early stopping condition 3 reached. Minimum error reduction."
			return self.create_leaf(target_values)

		remaining_features.remove(splitting_feature)
		if verbose: print "Split on feature %s. (%s, %s)" % (splitting_feature, len(left_split), len(right_split))

		# Repeat (recurse) on left and right subtrees
		left_tree = self.decision_tree_create(left_split, remaining_features, target,
			current_depth + 1, max_depth, min_node_size, min_error_reduction, verbose)

		right_tree = self.decision_tree_create(right_split, remaining_features, target,
			current_depth + 1, max_depth, min_node_size, min_error_reduction, verbose)

		leaf = {'is_leaf':False, 'prediction':None,
				'splitting_feature':splitting_feature,
				'left':left_tree,
				'right':right_tree}

		return leaf

	def count_nodes(self, tree):
		if tree['is_leaf']:
			return 1
		return 1 + self.count_nodes(tree['left']) + self.count_nodes(tree['right'])

	def count_leaves(self, tree):
		if tree['is_leaf']:
			return 1
		return self.count_leaves(tree['left']) + self.count_leaves(tree['right'])

	def classify(self, tree, x, verbose=False):
		# if the node is a leaf node.
		if tree['is_leaf']:
			if verbose:
				print "At leaf, predicting %s" % tree['prediction']
			return tree['prediction']
		else:
			# split on feature.
			split_feature_value = x[tree['splitting_feature']]
			if verbose:
				print "Split on %s = %s" % (tree['splitting_feature'],split_feature_value)
			if split_feature_value == 0:
				#print "--> left"
				return self.classify(tree['left'],x,verbose)
			else:
				# print "--> right"
				return self.classify(tree['right'],x,verbose)

	def evaluate_classification_error(self, tree, data, target):
		# Apply the classify(tree, x) to each row in your data
		prediction = data.apply(lambda x:self.classify(tree,x))

		# Once you've made the predictions, calculate the classification error and return it
		n_data = float(len(data))
		num_correct = (prediction == data[target]).sum()
		num_mistakes = n_data - num_correct

		classification_error = num_mistakes / n_data

		return classification_error

	def print_stump(self, tree,name='root'):
		split_name = tree['splitting_feature'] # split_name is something like 'term. 36 months'
		if split_name is None:
			print "(leaf, label: %s)" % tree['prediction']
			return None
		split_feature,split_value = split_name.split('.')
		print '                       %s' % name
		print '         |---------------|----------------|'
		print '         |                                |'
		print '         |                                |'
		print '         |                                |'
		print '  [{0} == 0]               [{0} == 1]    '.format(split_name)
		print '         |                                |'
		print '         |                                |'
		print '         |                                |'
		print '    (%s)                         (%s)' % (
		('leaf, label: ' + str(tree['left']['prediction']) if tree['left']['is_leaf'] else 'subtree'),
		('leaf, label: ' + str(tree['right']['prediction']) if tree['right']['is_leaf'] else 'subtree'))

	def reached_minimum_node_size(self, data,min_node_size):
		# Return True if the number of data points is less than or equal to the minimum node size.
		if len(data) <= min_node_size:
			return True
		return False

	def error_reduction(self, error_before_split,error_after_split):
		# Return the error before the split minus the error after the split.
		return (error_before_split - error_after_split)

class AdaBoost(DecisionTree):

	def __init__(self):
		super(AdaBoost, self).__init__()
		self.__name__ = 'adaboost_tree_model'

	def best_splitting_feature_weighted(self, data, features, target, data_weights):
		# These variables will keep track of the best feature and the corresponding error
		best_feature = None
		best_error = float('+inf')
		num_points = float(len(data))

		# Loop through each feature to consider splitting on that feature
		for feature in features:
			# The left split will have all data points where the feature value is 0
			# The right split will have all data points where the feature value is 1
			left_split = data[data[feature] == 0]
			right_split = data[data[feature] == 1]

			# Apply the same filtering to data_weights to create left_data_weights, right_data_weights
			left_data_weights = data_weights[data[feature] == 0]
			right_data_weights = data_weights[data[feature] == 1]

			# Calculate the weight of mistakes for left and right sides
			left_weighted_mistakes,left_class = self.intermediate_node_weighted_mistakes(left_split[target],
				left_data_weights)
			right_weighted_mistakes,right_class = self.intermediate_node_weighted_mistakes(right_split[target],
				right_data_weights)

			# Compute weighted error by computing
			#  ( [weight of mistakes (left)] + [weight of mistakes (right)] ) / [total weight of all data points]
			error = (left_weighted_mistakes + right_weighted_mistakes) / num_points

			# If this is the best error we have found so far, store the feature and the error
			if error < best_error:
				best_feature = feature
				best_error = error

		# Return the best feature we found
		return best_feature

	def intermediate_node_weighted_mistakes(self, labels_in_node,data_weights):
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
		if weighted_mistakes_all_negative < weighted_mistakes_all_positive:
			return (weighted_mistakes_all_negative,-1)
		else:
			return (weighted_mistakes_all_positive,+1)

	def create_leaf_weighted(self, target_values, data_weights):
		# Create a leaf node
		leaf = {'splitting_feature':None,'is_leaf':True}

		# Computed weight of mistakes.
		weighted_error,best_class = self.intermediate_node_weighted_mistakes(target_values,data_weights)

		# Store the predicted class (1 or -1) in leaf['prediction']
		leaf['prediction'] = 1 if best_class == 1 else -1

		return leaf

	def weighted_decision_tree_create(self, data, features, target, data_weights,
										current_depth=1,max_depth=10,verbose=True):

		remaining_features = features[:] # Make a copy of the features.
		target_values = data[target]
		if verbose:
			print "--------------------------------------------------------------------"
			print "Subtree, depth = %s (%s data points)." % (current_depth,len(target_values))

		# Stopping condition 1. Error is 0.
		if self.intermediate_node_weighted_mistakes(target_values,data_weights)[0] <= 1e-15:
			if verbose: print "Stopping condition 1 reached."
			return self.create_leaf_weighted(target_values,data_weights)

		# Stopping condition 2. No more features.
		if remaining_features == []:
			if verbose: print "Stopping condition 2 reached."
			return self.create_leaf_weighted(target_values,data_weights)

		# Additional stopping condition (limit tree depth)
		if current_depth > max_depth:
			if verbose: print "Reached maximum depth. Stopping for now."
			return self.create_leaf_weighted(target_values,data_weights)

		splitting_feature = self.best_splitting_feature_weighted(data,features,target,data_weights)
		remaining_features.remove(splitting_feature)

		left_split = data[data[splitting_feature] == 0]
		right_split = data[data[splitting_feature] == 1]

		left_data_weights = data_weights[data[splitting_feature] == 0]
		right_data_weights = data_weights[data[splitting_feature] == 1]

		if verbose:
			print "Split on feature %s. (%s, %s)" % (splitting_feature, len(left_split), len(right_split))

		# Create a leaf node if the split is "perfect"
		if len(left_split) == len(data):
			if verbose: print "Creating leaf node."
			return self.create_leaf_weighted(left_split[target], data_weights)
		if len(right_split) == len(data):
			if verbose: print "Creating leaf node."
			return self.create_leaf_weighted(right_split[target], data_weights)

		# Repeat (recurse) on left and right subtrees
		left_tree = self.weighted_decision_tree_create(
			left_split, remaining_features, target, left_data_weights, current_depth + 1, max_depth, verbose=verbose)
		right_tree = self.weighted_decision_tree_create(
			right_split, remaining_features, target, right_data_weights, current_depth + 1, max_depth, verbose=verbose)

		leaf = {'is_leaf'          : False,
				'prediction'       : None,
				'splitting_feature': splitting_feature,
				'left'             : left_tree,
				'right'            : right_tree}

		return leaf

	def adaboost_with_tree_stumps(self, data, features, target, num_tree_stumps, verbose=True):
		# start with unweighted data
		alpha = gp.graphlab.SArray([1.] * len(data))
		weights = []
		tree_stumps = []
		target_values = data[target]

		for t in xrange(num_tree_stumps):
			if verbose:
				print '====================================================='
				print 'Adaboost Iteration %d' % t
				print '====================================================='
			# Learn a weighted decision tree stump. Use max_depth=1
			tree_stump = self.weighted_decision_tree_create(data,features,target,data_weights=alpha,
															max_depth=1, verbose=verbose)
			tree_stumps.append(tree_stump)

			# Make predictions
			predictions = data.apply(lambda x:self.classify(tree_stump,x))

			# Produce a Boolean array indicating whether
			# each data point was correctly classified
			is_correct = predictions == target_values
			is_wrong = predictions != target_values

			# Compute weighted error
			#weighted_error = is_wrong.sum()/float(is_correct.sum()+is_wrong.sum())
			weighted_error = alpha[is_wrong].sum() / float(alpha.sum())

			# Compute model coefficient using weighted error
			weight = (1 / 2.0) * log((1 - weighted_error) / weighted_error)
			weights.append(weight)

			# Adjust weights on data point
			adjustment = is_correct.apply(lambda is_correct:exp(-weight) if is_correct else exp(weight))

			# Scale alpha by multiplying by adjustment
			# Then normalize data points weights
			alpha = alpha * adjustment
			alpha = alpha / float(sum(alpha))

		return weights,tree_stumps

	def predict_adaboost(self, stump_weights,tree_stumps,data):
		scores = gp.graphlab.SArray([0.] * len(data))

		for i,tree_stump in enumerate(tree_stumps):
			predictions = data.apply(lambda x:self.classify(tree_stump,x))

			# Accumulate predictions on scores array
			scores = scores + (stump_weights[i] * predictions)

		return scores.apply(lambda score:+1 if score > 0 else -1)

	def evaluate_classification_error_weighted(self,tree,data,target):
		# Apply the classify(tree, x) to each row in your data
		prediction = data.apply(lambda x:self.classify(tree,x))

		# Once you've made the predictions, calculate the classification error
		return (prediction != data[target]).sum() / float(len(data))

class LogisticRregStochastic(object):
	def __init__(self):
		self.__name__="logistic_regression_stochastic_model"

	def logistic_regression_SG(self, feature_matrix,sentiment,initial_coefficients,step_size,batch_size,max_iter,verbose=True):
		log_likelihood_all = []

		# make sure it's a numpy array
		coefficients = np_utils.np.array(initial_coefficients)
		# set seed=1 to produce consistent results
		np_utils.np.random.seed(seed=1)
		# Shuffle the data before starting
		permutation = np_utils.np.random.permutation(len(feature_matrix))
		feature_matrix = feature_matrix[permutation,:]
		sentiment = sentiment[permutation]

		i = 0 # index of current batch
		# Do a linear scan over data
		for itr in xrange(max_iter):
			# Predict P(y_i = +1|x_i,w) using your predict_probability() function
			# Make sure to slice the i-th row of feature_matrix with [i:i+batch_size,:]
			predictions = np_utils.predict_probability(feature_matrix[i:i + batch_size,:],coefficients)

			# Compute indicator value for (y_i = +1)
			# Make sure to slice the i-th entry with [i:i+batch_size]
			indicator = (sentiment[i:i + batch_size] == +1)

			# Compute the errors as indicator - predictions
			errors = indicator - predictions
			for j in xrange(len(coefficients)): # loop over each coefficient
				# Recall that feature_matrix[:,j] is the feature column associated with coefficients[j]
				# Compute the derivative for coefficients[j] and save it to derivative.
				# Make sure to slice the i-th row of feature_matrix with [i:i+batch_size,j]
				derivative = np_utils.feature_derivative(errors,feature_matrix[i:i + batch_size,j])

				# compute the product of the step size, the derivative, and the **normalization constant** (1./batch_size)
				coefficients[j] += step_size * derivative * (1. / batch_size)

			# Checking whether log likelihood is increasing
			# Print the log likelihood over the *current batch*
			lp = np_utils.compute_avg_log_likelihood(feature_matrix[i:i + batch_size,:],sentiment[i:i + batch_size],coefficients)
			log_likelihood_all.append(lp)
			if verbose:
				if itr <= 15 or (itr <= 1000 and itr % 100 == 0) or (
						itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0 or itr == max_iter - 1:
					data_size = len(feature_matrix)
					print 'Iteration %*d: Average log likelihood (of data points in batch [%0*d:%0*d]) = %.8f' % (
					int(np_utils.np.ceil(np_utils.np.log10(max_iter))),itr,int(np_utils.np.ceil(np_utils.np.log10(data_size))), i,\
						      int(np_utils.np.ceil(np_utils.np.log10(data_size))), i+ batch_size,lp)

			# if we made a complete pass over data, shuffle and restart
			i += batch_size
			if i + batch_size > len(feature_matrix):
				permutation = np_utils.np.random.permutation(len(feature_matrix))
				feature_matrix = feature_matrix[permutation,:]
				sentiment = sentiment[permutation]
				i = 0

		# We return the list of log likelihoods for plotting purposes.
		return coefficients,log_likelihood_all
#==================================================================
#                   Helper Functions
#==================================================================
def get_model_classification_accuracy(model,data,true_labels):
	""" e.g model = sentiment_model,
			data = test_data,
			labels = test_data['sentiment'] (+1,-1,-1....)
		accuracy = # correctly classified examples / total examples"""

	# First get the predictions
	predictions = model.predict(data)
	# Compute the number of correctly classified examples
	corrected_classified_labels = (predictions == true_labels).sum()
	# Then compute accuracy by dividing num_correct by total number of examples
	total_examples = float(data.num_rows())
	accuracy = corrected_classified_labels / total_examples

	return accuracy

def get_majority_class_accuracy(model, data_set):
	""" majority class -> sentiment == +1 (positive classify)"""
	majority_class = (model.predict(data_set) == +1).sum()
	n_total_data = float(data_set.num_rows())
	accuracy = majority_class/n_total_data
	return accuracy

def get_classification_accuracy(feature_matrix,sentiment,coefficients):
	scores = np_utils.np.dot(feature_matrix,coefficients)
	apply_threshold = np_utils.np.vectorize(lambda x:1. if x > 0  else -1.)
	predictions = apply_threshold(scores)
	num_correct = (predictions == sentiment).sum()
	accuracy = num_correct / float(len(feature_matrix))
	return accuracy

def get_classification_error(model, dataset):
	accuracy = model.evaluate(dataset)['accuracy']
	return (1-accuracy)

def get_training_errors(model_n, dataset, model_name):
	training_errors = []
	for n_iterations in model_name:
		accuracy_n = model_n[n_iterations].evaluate(dataset)['accuracy']
		training_errors.append(1-accuracy_n)
	return training_errors

def apply_threshold(probabilities, threshold):
	# +1 if >= threshold and -1 otherwise.
	return probabilities.apply(lambda prob: +1 if prob >= threshold else -1)