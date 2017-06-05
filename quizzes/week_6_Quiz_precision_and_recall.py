__author__ = 'nadyaK'
__date__ = '05/28/2017'

import ml_graphlab_utils as gp
import ml_classification_utils as cl_utils
import ml_numpy_utils as np_utils
import ml_plotting_utils as np_plot
import traceback

#==================================================================
#              Quiz-1: Precision & Recall (curve)
#==================================================================
def precision_and_recall_plot(model,test_data):
	threshold_values = np_utils.np.linspace(0.5,1,num=100)
	probabilities = model.predict(test_data,output_type='probability')
	precision_all,recall_all = get_all_precisions_and_recall(test_data,probabilities,threshold_values)
	output_file = '../graphs/Precision_recall_curve.png'
	np_plot.plot_pr_curve(precision_all,recall_all,'Precision recall curve (all)',output_file)
	threshold_small = find_smallest_threshold(precision_all,threshold_values)
	print "\nQ10: smallest threshold value that achieves a precision of 96.5 or better is: %s" % round(threshold_small,
		3)
	predictions_high = cl_utils.apply_threshold(probabilities,0.98)
	conf_matrix = gp.graphlab.evaluation.confusion_matrix(test_data['sentiment'],predictions_high)
	print conf_matrix
	print "\nQ11: False negatives (+1)(-1) in test-data: 5826"

def get_all_precisions_and_recall(dataset,probabilities,thresholds_values):
	precision_all = []
	recall_all = []

	for threshold in thresholds_values:
		predictions = cl_utils.apply_threshold(probabilities,threshold)

		precision = gp.graphlab.evaluation.precision(dataset['sentiment'],predictions)
		recall = gp.graphlab.evaluation.recall(dataset['sentiment'],predictions)

		precision_all.append(precision)
		recall_all.append(recall)

	return precision_all,recall_all

def precision_and_recall_threshold(model,test_data):
	probabilities = model.predict(test_data,output_type='probability')
	predictions_with_default_threshold = cl_utils.apply_threshold(probabilities,0.5)
	predictions_with_high_threshold = cl_utils.apply_threshold(probabilities,0.9)
	print "\nQ8:  as the threshold increased from 0.5 to 0.9 the number of positive predicted reviews: decreased"
	print "\t(threshold = 0.5): %s" % (predictions_with_default_threshold == 1).sum()
	print "\t(threshold = 0.9): %s" % (predictions_with_high_threshold == 1).sum()

	# Threshold = 0.5
	precision_with_default_threshold = gp.graphlab.evaluation.precision(test_data['sentiment'],
		predictions_with_default_threshold)

	recall_with_default_threshold = gp.graphlab.evaluation.recall(test_data['sentiment'],
		predictions_with_default_threshold)

	# Threshold = 0.9
	precision_with_high_threshold = gp.graphlab.evaluation.precision(test_data['sentiment'],
		predictions_with_high_threshold)
	recall_with_high_threshold = gp.graphlab.evaluation.recall(test_data['sentiment'],predictions_with_high_threshold)

	print "\nQ9:  YES, precision increase with higher threshold"
	print "\t(threshold = 0.5): %s" % precision_with_default_threshold
	print "\t(threshold = 0.9): %s" % precision_with_high_threshold

	print "\nQ9-2:  FALSE, recall does not increase with higher threshold"
	print "\t(threshold = 0.5): %s" % recall_with_default_threshold
	print "\t(threshold = 0.9): %s" % recall_with_high_threshold

def find_smallest_threshold(precision_all, threshold_values):
	precision_above96percent = filter(lambda precision: precision >= 0.965, precision_all)
	idx_precision = precision_all.index(min(precision_above96percent))
	return threshold_values[idx_precision]

def main():
	try:
		print "\n**************************************"
		print "*        Precision & Recall          *"
		print "**************************************\n"

		products = gp.load_data('../../data_sets/amazon_baby.gl/')

		# Remove punctuation.
		review_clean = products['review'].apply(gp.remove_punctuation)

		# Count words
		products['word_count'] = gp.graphlab.text_analytics.count_words(review_clean)

		# Drop neutral sentiment reviews.
		products = products[products['rating'] != 3]

		# Positive sentiment to +1 and negative sentiment to -1
		products['sentiment'] = products['rating'].apply(lambda rating:+1 if rating > 3 else -1)
		train_data,test_data = products.random_split(.8,seed=1)
		model = gp.graphlab.logistic_classifier.create(train_data, target='sentiment',
											features=['word_count'],
											validation_set=None, verbose=False)
		accuracy = model.evaluate(test_data,metric='accuracy')['accuracy']
		baseline = len(test_data[test_data['sentiment'] == 1]) / len(test_data)

		print "\nQ1: YES, logistic regression model was better than the baseline (majority class classifier)"
		print "\tBaseline: %s" % accuracy
		print "\tReg-model: %s" % baseline
		confusion_matrix = model.evaluate(test_data,metric='confusion_matrix')['confusion_matrix']
		print confusion_matrix
		false_positives = 1443
		print "\nQ2: False positives: (-1)(+1): %s" % false_positives
		false_negatives = 1406
		print "\nQ3: Cost associated with the logistic regression: $%s" % (false_negatives + 100 * false_positives)
		true_positives = 26689
		print "\nQ4: Fracion of false positives: %s" % round((false_positives / float(true_positives)),2)
		print "\nQ5: Increase threshold for predicting the positive class (y^=+1)"
		print "\nQ6: Fracion of false positives: %s" % round((false_negatives / float(true_positives)),2)
		print "\nQ7: classifier that predicts +1 for all data points has recall= 1"

		precision_and_recall_threshold(model,test_data)

		precision_and_recall_plot(model,test_data)

		baby_reviews = test_data[test_data['name'].apply(lambda x:'baby' in x.lower())]
		probabilities_baby = model.predict(baby_reviews,output_type='probability')
		threshold_values_baby = np_utils.np.linspace(0.5,1,num=100)
		precision_all_baby,recall_all_baby = get_all_precisions_and_recall(baby_reviews,probabilities_baby,
			threshold_values_baby)
		threshold_small_baby = find_smallest_threshold(precision_all_baby,threshold_values_baby)
		print "\nQ12: smallest threshold-baby value that achieves a precision of 96.5 or better is: %s" % round(
			threshold_small_baby,3)
		print "\nQ13: threshold value is larger: than the threshold used for the entire dataset"
		output_file = '../graphs/Precision_recall_curve_baby.png'
		np_plot.plot_pr_curve(precision_all_baby,recall_all_baby,"Precision-Recall (Baby)",output_file)

	except Exception as details:
		print (">> Exit or Errors \n%s, %s"%(details, traceback.print_exc()))

if __name__ == "__main__":
	main()