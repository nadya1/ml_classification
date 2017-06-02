__author__ = 'nadyaK'
__date__ = '05/27/2017'

import ml_graphlab_utils as gp
import ml_classification_utils as cl_utils
import traceback
 
#==================================================================
#              Quiz-1: Early Stop (decision tree)
#==================================================================
def select_small_features():
	# Features for the classification algorithm
	features = ['grade',# grade of the loan
		'term',# the term of the loan
		'home_ownership',# home_ownership status: own, mortgage or rent
		'emp_length',# number of years of employment
		]
	return features

def evaluate_classification_error_per_tree(num_models, data_set, models, dt_class):
	for n_model in num_models:
		print "\n\tmodel_%s:" %(n_model)
		print "\t\tclassification error:", dt_class.evaluate_classification_error(models['model_%s'%n_model], data_set, 'safe_loans')

def count_leaves_per_model(num_models, dt_class, models):
	for n_model in num_models:
		print "\n\tmodel_%s:" %(n_model)
		print "\n\tCount leaves:", dt_class.count_leaves(models['model_%s'%n_model])

def quiz1_early_stopping(loans, target):
	small_features = select_small_features()
	loans_small = loans[small_features + [target]]
	loans_data = gp.subsample_dataset_to_balance_classes(loans_small,target)

	loans_data = gp.transform_categorical_into_bin_features(loans_data,small_features)

	features = loans_data.column_names()
	features.remove('safe_loans')  # Remove the response variable
	# print features
	train_data,validation_set = loans_data.random_split(.8,seed=1)

	dt_class = cl_utils.DecisionTree()

	small_sample = gp.graphlab.SArray([1,1,1,1,1,1,-1,-1,-1])
	reachmin = dt_class.reached_minimum_node_size(small_sample,min_node_size=10)
	print "\nQ1: with 6 safe loans and 3 risky loans, if the min_node_size parameter is 10, reached MIN-NODE: %s" \
		  "\n\t->Create a leaf and return it " %reachmin

	# error_before_split = 0.2
	# for errors in [0.0,0.5,0.1,0.14]:
	# 	print 'error reduction: %s'%dt_class.error_reduction(error_before_split, errors)
	print "\nQ2: error_reduction=0.2 n->Create a leaf and return it"
	my_decision_tree_old = dt_class.decision_tree_create(train_data, features, 'safe_loans', max_depth = 6,
									min_node_size = 0, min_error_reduction=-1, verbose = False)
	my_decision_tree_new = dt_class.decision_tree_create(train_data, features, 'safe_loans', max_depth = 6,
									min_node_size = 100, min_error_reduction=0.0, verbose = False)

	print "\nQ3: Prediction path shorter is using my_decision_tree_new?"
	print '\tOld:', dt_class.classify(my_decision_tree_old, validation_set[0], verbose = True)
	print '\tNew:', dt_class.classify(my_decision_tree_new, validation_set[0], verbose = True)

	print "\nQ4: For a data point using my_decision_tree_new, prediction path is: Shorter or the same"
	print "\nQ5: For a tree trained on any dataset using max_depth = 6 maximum num of splits is: 6"

	new_error = dt_class.evaluate_classification_error(my_decision_tree_new, validation_set, target)
	old_error = dt_class.evaluate_classification_error(my_decision_tree_old, validation_set, target)

	print "\nQ6: error of the new-decision tree is lower than old-decision tree: %s vs %s" % (new_error,old_error )

	print "\nBuilding Decision Trees models 1-2-3 ...."
	models = {}
	for n_model,depth in [(1,2),(2,6),(3,14)]:
		print "\tmodel_%s, depth:%s" % (n_model,depth)
		models['model_%s' % n_model] = dt_class.decision_tree_create(train_data,features,'safe_loans',max_depth=depth,
										min_node_size=0,min_error_reduction=-1, verbose=False)
	# print "TRAINING DATA:"
	# evaluate_classification_error_per_tree([1,2,3], train_data, models, dt_class)
	# print "VALIDATION DATA:"
	# evaluate_classification_error_per_tree([1,2,3], validation_set, models, dt_class)

	print "\nQ7: Model_3 has smallest error on the validation data: 0.380008616975"
	print "\nQ8: YES, smallest error in the training data also have the smallest error in the validation data"
	print "\nQ9: No, this is NOT ALWAYS true. It depends on the data and max-depth"
	count_leaves_per_model([1,2,3], dt_class, models)
	print "\nQ10: Model_3 tree has the largest complexity: 341 leaves"
	print "\nQ11: FALSE, Not always most complex tree will result in the lowest classification error, it may be overfitting"

	for n_model,error_eval in [(4,-1),(5,0),(6,5)]:
		print "\tmodel_%s, error_reduction:%s" %(n_model,error_eval)
		models['model_%s'%n_model] = dt_class.decision_tree_create(train_data, features, 'safe_loans', max_depth = 6,
									min_node_size = 0, min_error_reduction = error_eval, verbose=False)

	print "\nQ12: Model_4 tree has the largest complexity: with 41 leaves"
	evaluate_classification_error_per_tree([4,5,6],validation_set,models,dt_class)
	print "\nQ13: YES, Pick model_5 over model_4"

	for n_model,node_change_size in [(7,0),(8,2000),(9,50000)]:
		print "\tmodel_%s, node_change_size:%s" % (n_model,node_change_size)
		models['model_%s' % n_model] = dt_class.decision_tree_create(train_data,features,'safe_loans',max_depth=6,
									min_node_size=node_change_size,min_error_reduction=-1, verbose=False)

	# count_leaves_per_model([7,8,9],dt_class, models)
	print "\nQ14: Model 8: count leaves: 19"

def main():
	try:
		print "\n**************************************"
		print "*          Decision Trees            *"
		print "**************************************\n"
		print "Building Decision Tree: ..."

		loans = gp.load_data('../../data_sets/lending-club-data.gl/')

		# Remove bad_loands column
		loans['safe_loans'] = loans['bad_loans'].apply(lambda x:+1 if x == 0 else -1)
		loans = loans.remove_column('bad_loans')

		# Extract the feature columns and target column
		target = 'safe_loans' # prediction target (y) (+1 means safe, -1 is risky)

		quiz1_early_stopping(loans,target)


	except Exception as details:
		print (">> Exit or Errors \n%s, %s"%(details, traceback.print_exc()))

if __name__ == "__main__":
	main()