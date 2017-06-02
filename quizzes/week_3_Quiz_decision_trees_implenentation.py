__author__ = 'nadyaK'
__date__ = '05/24/2017'

import ml_graphlab_utils as gp
import ml_classification_utils as cl_utils
import ml_numpy_utils as np_utils
import matplotlib.pyplot as plt
import traceback

#==================================================================
#                 Quiz-1: Decision Tree
#==================================================================
def quiz1_identify_safe_loans(loans, target):
	# Decision tree to build a classifier
	features = select_features()
	loans = loans[features + [target]]

	loans_data = gp.subsample_dataset_to_balance_classes(loans,target)
	train_data,validation_data = loans_data.random_split(.8,seed=1)


	decision_tree_model = gp.graphlab.decision_tree_classifier.create(train_data,validation_set=None,target=target,
		features=features,verbose=False)
	# Depth 2 or 10
	small_model = gp.graphlab.decision_tree_classifier.create(train_data,validation_set=None,target=target,
		features=features,max_depth=2,verbose=False)
	big_model = gp.graphlab.decision_tree_classifier.create(train_data,validation_set=None,target=target,features=features,
		max_depth=10,verbose=False)

	sample_validation_data = get_simple_val_data(validation_data,target)
	predictions = decision_tree_model.predict(sample_validation_data)#[target_idx+1]
	correct_classified = list(set(predictions) & set(sample_validation_data['safe_loans']))
	print "\nQ1: correct_classified (%s) is: %%%s" % (correct_classified,100 * (len(correct_classified) / 4.0))

	probability_predict = decision_tree_model.predict(sample_validation_data,output_type='probability')
	# print probability_predict
	print "\nQ2: Loan that has highest probability of being classified as a safe loan is 4th load"

	prob_small_predict = small_model.predict(sample_validation_data,output_type='probability')
	# print prob_small_predict
	print "\nQ3: During tree traversal both examples fall into the same leaf node."

	# small_model.show(view="Tree")
	print "\nQ4: Based on the visualized tree, prediction will be -1."

	print "\nQ5: accuracy of decision_tree_model on the validation set is: %s" % round(
		decision_tree_model.evaluate(validation_data)['accuracy'],2)

	print "\nQ6: big_model perform worst than decision_tree_model and it is a sign of overfitting"
	print '\tbig-model(val): ',big_model.evaluate(validation_data)['accuracy']
	print '\tdecision-(val): ',decision_tree_model.evaluate(validation_data)['accuracy']

	predictions_val = decision_tree_model.predict(validation_data)

	false_positives = np_utils.compute_false_positive(validation_data['safe_loans'],predictions_val)
	false_negatives = np_utils.compute_false_negative(validation_data['safe_loans'],predictions_val)
	cost_of_mistakes = (false_positives * 20000) + (false_negatives * 10000)
	print "\nQ7: total cost of mistakes made by decision_tree_model on validation_data is: %s" % cost_of_mistakes
	print "\tTotal false positives is: %s" % false_positives
	print "\tTotal false negatives is: %s" % false_negatives

def select_features():
	# Features for the classification algorithm
	features = ['grade',# grade of the loan
		'sub_grade',# sub-grade of the loan
		'short_emp',# one year or less of employment
		'emp_length_num',# number of years of employment
		'home_ownership',# home_ownership status: own, mortgage or rent
		'dti',# debt to income ratio
		'purpose',# the purpose of the loan
		'term',# the term of the loan
		'last_delinq_none',# has borrower had a delinquincy
		'last_major_derog_none',# has borrower had 90 day or worse rating
		'revol_util',# percent of available credit being used
		'total_rec_late_fee',# total late fees received to day]
		]
	return features

def get_simple_val_data(validation_data, target):
	validation_safe_loans = validation_data[validation_data[target] == 1]
	validation_risky_loans = validation_data[validation_data[target] == -1]

	sample_validation_data_risky = validation_risky_loans[0:2]
	sample_validation_data_safe = validation_safe_loans[0:2]

	sample_validation_data = sample_validation_data_safe.append(sample_validation_data_risky)

	return sample_validation_data

#==================================================================
#                 Quiz-2: L2 regularization
#==================================================================
def select_small_features():
	# Features for the classification algorithm
	features = ['grade',# grade of the loan
		'term',# the term of the loan
		'home_ownership',# home_ownership status: own, mortgage or rent
		'emp_length',# number of years of employment
		]
	return features

def quiz2_buildng_decision_tree(loans, target):
	small_features = select_small_features()
	loans_small = loans[small_features + [target]]
	loans_data = gp.subsample_dataset_to_balance_classes(loans_small,target)

	loans_data = gp.transform_categorical_into_bin_features(loans_data,small_features)

	features = loans_data.column_names()
	features.remove('safe_loans')  # Remove the response variable
	# print features
	train_data,test_data = loans_data.random_split(.8,seed=1)

	dt_class = cl_utils.DecisionTree()

	# Make sure to cap the depth at 6 by using max_depth = 6
	my_decision_tree = dt_class.decision_tree_create(train_data,features,'safe_loans',max_depth=6.,verbose=False)

	dt_class.classify(my_decision_tree,test_data[0],verbose=True)

	print "\nQ1: First split is: term. 36 months "
	print "\nQ2: First feature that lead to a right split is: Split on grade.D "
	print "\nQ3: Last feature split on before reaching a leaf node is: Split on grade.D "

	classification_error = dt_class.evaluate_classification_error(my_decision_tree,test_data,target)
	print "\nQ4: classification error of my_decision_tree on the test_data is: %s" % round(classification_error,2)
	dt_class.print_stump(my_decision_tree)

	#Explore subtree
	dt_class.print_stump(my_decision_tree['left']['left'],my_decision_tree['left']['splitting_feature'])
	print "\nQ6: first 3 feature splits considered along the left-most branch are:\n\t--> term. 36 months, grade.A == 0, grade.B == 0"

	dt_class.print_stump(my_decision_tree['right'],my_decision_tree['splitting_feature'])
	print "\nQ7: first 3 feature splits considered along the right-most  branch are:\n\t--> term. 36 months, grade.D == 0, leaf, label: -1 "


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

		quiz1_identify_safe_loans(loans,target)

		quiz2_buildng_decision_tree(loans,target)

	except Exception as details:
		print (">> Exit or Errors \n%s, %s"%(details, traceback.print_exc()))

if __name__ == "__main__":
	main()