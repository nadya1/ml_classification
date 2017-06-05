__author__ = 'nadyaK'
__date__ = '05/28/2017'

import ml_graphlab_utils as gp
import ml_classification_utils as cl_utils
import ml_numpy_utils as np_utils
import ml_plotting_utils as np_plot 
import traceback
 
#==================================================================
#              Quiz-1: Early Stop (decision tree)
#==================================================================
def quiz1_boosting_trees(loans, target):
	features = select_features()
	loans,loans_with_na = loans[[target] + features].dropna_split()
	# num_rows_with_na = loans_with_na.num_rows()
	# num_rows = loans.num_rows()
	# print 'Dropping %s observations; keeping %s ' % (num_rows_with_na,num_rows)
	loans_data = gp.subsample_dataset_to_balance_classes(loans,target)

	train_data,validation_data = loans_data.random_split(.8,seed=1)
	model_5 = gp.graphlab.boosted_trees_classifier.create(train_data,validation_set=None,
						target=target,features=features,max_iterations=5, verbose=False)
	sample_validation_data = small_val_data(validation_data, target)
	predictions = model_5.predict(sample_validation_data)
	# print predictions
	print "\nQ1: percentage: %s%%" % cl_utils.get_model_classification_accuracy(model_5,sample_validation_data, sample_validation_data['safe_loans'])

	percent_pred = model_5.predict(sample_validation_data,output_type='probability')
	# print percent_pred
	print "\nQ2: Loan that is least likely to be safe loan: %sth" % (percent_pred.argmin() + 1)
	actual_labels = validation_data['safe_loans']
	predictions_labels = model_5.predict(validation_data)
	false_positive = np_utils.compute_false_positive(actual_labels,predictions_labels)
	print "\nQ3: False positives on validation data :%s" % false_positive
	false_negative = np_utils.compute_false_negative(actual_labels,predictions_labels)
	# print "\nQ3-b: False negative on validation data :%s" % false_negative
	print "\nQ4: Cost boosted tree: $%s" % (10000 * false_negative + 20000 * false_positive)
	prob_model = model_5.predict(validation_data,output_type='probability')
	# print prob_model
	validation_data['predictions'] = prob_model
	
	safe_loans_probability = validation_data[validation_data['safe_loans'] == 1]
	top5_grades = safe_loans_probability.topk('predictions',k=5,reverse=False)['grade']
	top5_prob = safe_loans_probability.topk('predictions',k=5,reverse=False)['predictions']

	print "\nQ5: grades of the top5 safe loan: %s" % top5_grades

	print "\nBuilding Boosting Tree: ..."
	model_n = {}
	for n_iterations in [10,50,100,200,500]:
		model_n[n_iterations] = gp.graphlab.boosted_trees_classifier.create(train_data,validation_set=None,target=target,
			features=features,max_iterations=n_iterations,verbose=False)

	print "\nQ6: best accuracy (model_100): 0.691727703576"
	for n_iterations in [10,50,100,200,500]:
		accuracy_n = model_n[n_iterations].evaluate(validation_data)['accuracy']
		print "\tAccuracy (model_%s): %s" % (n_iterations,accuracy_n)

	print "\nQ7: NO, it varies depends on the dataset"
	training_errors = cl_utils.get_training_errors(model_n,train_data,[10,50,100,200,500])
	validation_errors = cl_utils.get_training_errors(model_n,validation_data,[10,50,100,200,500])
	print "\nQ10: YES, training error reduces as the number of trees increases"
	print '\tTraining Errors: %s'%training_errors
	print '\tValidation Errors: %s'%validation_errors
	print "\nQ11: FALSE, it is not always true validation error will reduce as the number of trees increases"
	output_file = '../graphs/TrainErrors_vs_ValErrors.png'
	np_plot.make_train_error_vs_val_error_trees_plot(training_errors, validation_errors, output_file)
	
def small_val_data(validation_data, target):
	# Select all positive and negative examples.
	validation_safe_loans = validation_data[validation_data[target] == 1]
	validation_risky_loans = validation_data[validation_data[target] == -1]

	# Select 2 examples from the validation set for positive & negative loans
	sample_validation_data_risky = validation_risky_loans[0:2]
	sample_validation_data_safe = validation_safe_loans[0:2]

	# Append the 4 examples into a single dataset
	sample_validation_data = sample_validation_data_safe.append(sample_validation_data_risky)

	return sample_validation_data

def select_features():
	# Features for the classification algorithm
	features = ['grade',# grade of the loan (categorical)
		'sub_grade_num',# sub-grade of the loan as a number from 0 to 1
		'short_emp',# one year or less of employment
		'emp_length_num',# number of years of employment
		'home_ownership',# home_ownership status: own, mortgage or rent
		'dti',# debt to income ratio
		'purpose',# the purpose of the loan
		'payment_inc_ratio',# ratio of the monthly payment to income
		'delinq_2yrs',# number of delinquincies
		'delinq_2yrs_zero',# no delinquincies in last 2 years
		'inq_last_6mths',# number of creditor inquiries in last 6 months
		'last_delinq_none',# has borrower had a delinquincy
		'last_major_derog_none',# has borrower had 90 day or worse rating
		'open_acc',# number of open credit accounts
		'pub_rec',# number of derogatory public records
		'pub_rec_zero',# no derogatory public records
		'revol_util',# percent of available credit being used
		'total_rec_late_fee',# total late fees received to day
		'int_rate',# interest rate of the loan
		'total_rec_int',# interest received to date
		'annual_inc',# annual income of borrower
		'funded_amnt',# amount committed to the loan
		'funded_amnt_inv',# amount committed by investors for the loan
		'installment',# monthly payment owed by the borrower
		]
	return features
#==================================================================
#                 Quiz-2: Adaboosting
#==================================================================
def select_small_features():
	# Features for the classification algorithm
	features = ['grade',# grade of the loan
		'term',# the term of the loan
		'home_ownership',# home_ownership status: own, mortgage or rent
		'emp_length',# number of years of employment
		]
	return features

def mini_test(ada):
	example_labels = gp.graphlab.SArray([-1,-1,1,1,1])
	example_data_weights = gp.graphlab.SArray([1.,2.,.5,1.,1.])
	if ada.intermediate_node_weighted_mistakes(example_labels,example_data_weights) == (2.5,-1):
		print 'Test passed!'
	else:
		print 'Test failed... try again!'
def mini_test_split(ada,train_data, features, target):
	example_data_weights = gp.graphlab.SArray(len(train_data)* [1.5])
	if ada.best_splitting_feature_weighted(train_data, features, target, example_data_weights) == 'term. 36 months':
		print 'Test passed!'
	else:
		print 'Test failed... try again!'

def quiz2_adaboosting_trees(loans, target, verbose=False):
	ada = cl_utils.AdaBoost()
	features = select_small_features()
	loans,loans_with_na = loans[[target] + features].dropna_split()
	loans_data = gp.subsample_dataset_to_balance_classes(loans,target)
	loans_data = gp.transform_categorical_into_bin_features(loans_data,features)
	train_data,test_data = loans_data.random_split(0.8,seed=1)
	features = loans_data.column_names()
	features.remove('safe_loans')  # Remove the response variable
	# mini_test(ada)
	# mini_test_split(ada,train_data,features,target)

	print "\nQ1:  How is the weight of mistakes related to the classification error  WM(theta,y)= N * [classification error]"

	example_data_weights = gp.graphlab.SArray([1.0 for i in range(len(train_data))])
	small_data_decision_tree = ada.weighted_decision_tree_create(train_data,features,target,example_data_weights,
								max_depth=2,verbose=verbose)

	# Assign weights
	example_data_weights_sub = gp.graphlab.SArray([1.] * 10 + [0.] * (len(train_data) - 20) + [1.] * 10)

	# Train a weighted decision tree model.
	small_data_decision_tree_subset_20 = ada.weighted_decision_tree_create(train_data,features,target,example_data_weights_sub,
								max_depth=2,verbose=verbose)
	subset_20 = train_data.head(10).append(train_data.tail(10))
	print "\nQ2: YES, decision_tree_subset_20 same model as training 20 data points with non-zero weights"
	print "\tSubset: %s"%ada.evaluate_classification_error_weighted(small_data_decision_tree_subset_20, subset_20, target)
	print "\tTrain:  %s"%ada.evaluate_classification_error_weighted(small_data_decision_tree_subset_20, train_data,target)

	stump_weights,tree_stumps = ada.adaboost_with_tree_stumps(train_data,features,target,num_tree_stumps=10,verbose=verbose)
	predictions = ada.predict_adaboost(stump_weights, tree_stumps, test_data)
	# accuracy = gp.graphlab.evaluation.accuracy(test_data[target], predictions)
	print "\nQ3: component weights monotonically decreasing, monotonically increasing, or neither: NEITHER"
	print "\t: %s" % stump_weights
	stump_weights,tree_stumps = ada.adaboost_with_tree_stumps(train_data,features,target,num_tree_stumps=30,verbose=verbose)
	print "\nQ4: Training error goes down in general, with some ups and downs in the middle."
	error_all = []
	for n in xrange(1,31):
		predictions = ada.predict_adaboost(stump_weights[:n],tree_stumps[:n],train_data)
		error = 1.0 - gp.graphlab.evaluation.accuracy(train_data[target],predictions)
		error_all.append(error)
		# print "Iteration %s, training error = %s" % (n,error_all[n - 1])

	output_file = '../graphs/PerformanceAdaboost.png'
	np_plot.make_performance_of_adaboost_plot(error_all, output_file)

	test_error_all = []
	for n in xrange(1,31):
		predictions = ada.predict_adaboost(stump_weights[:n],tree_stumps[:n],test_data)
		error = 1.0 - gp.graphlab.evaluation.accuracy(test_data[target],predictions)
		test_error_all.append(error)
		# print "Iteration %s, test error = %s" % (n,test_error_all[n - 1])
	output_file = '../graphs/PerformanceAdaboost_error.png'
	np_plot.make_performance_of_adaboost_plot_error(error_all,test_error_all,output_file)
	print "\nQ5: plot (with 30 trees), is there massive overfitting as the # of iterations increases: FALSE"

def main():
	try:
		print "\n**************************************"
		print "*          Boosting Trees            *"
		print "**************************************\n"

		loans = gp.load_data('../../data_sets/lending-club-data.gl/')

		# Remove bad_loands column
		loans['safe_loans'] = loans['bad_loans'].apply(lambda x:+1 if x == 0 else -1)
		loans = loans.remove_column('bad_loans')

		# Extract the feature columns and target column
		target = 'safe_loans' # prediction target (y) (+1 means safe, -1 is risky)

		# quiz1_boosting_trees(loans,target)
		quiz2_adaboosting_trees(loans,target)

	except Exception as details:
		print (">> Exit or Errors \n%s, %s"%(details, traceback.print_exc()))

if __name__ == "__main__":
	main()