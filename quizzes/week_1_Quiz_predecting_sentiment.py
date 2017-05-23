__author__ = 'nadyaK'
__date__ = '05/12/2017'

import ml_graphlab_utils as gp
import ml_classification_utils as cl_utils
import ml_numpy_utils as np_utils
import traceback

def predict_scores_simple_data(model, test_data):
	"""Predict scores in simple test-inputs (test_data[10:13])"""
	sample_test_data = test_data[10:13]
	scores = model.predict(sample_test_data,output_type='margin')
	my_predictions = np_utils.sigmoid_function(scores)
	# print '\nScores: (output_type=margin) %s'%scores
	# print 'Scores: (output_type=probability) %s'%model.predict(sample_test_data, output_type='probability')
	# print "My predictions: %s" % my_predictions
	return my_predictions

def find_top_positive_reviews(test_data):
	top_20_positive = test_data.topk("probability",k=20,reverse=False)['name']
	found_names = []
	for name in top_20_positive:
		quiz_names = ['Snuza Portable Baby Movement Monitor','MamaDoo Kids Foldable Play Yard Mattress Topper,blue',
			'Britax Decathlon Convertible Car Seat, Tiffany','Safety 1st Exchangeable Tip 3 in 1 Thermometer']
		if name in quiz_names:
			found_names.append(name)
	return found_names

def find_top_negative_reviews(test_data):
	top_20_negative = test_data.topk("probability",k=20,reverse=True)['name']
	found_names = []
	for name in top_20_negative:
		quiz_names = ['The First Years True Choice P400 Premium Digital Monitor, 2 Parent Unit',
			'JP Lizzy Chocolate Ice Classic Tote Set','Peg-Perego Tatamia High Chair, White Latte',
			'Safety 1st High-Def Digital Monitor']
		if name in quiz_names:
			found_names.append(name)
	return found_names

def quiz1_make_predictions_logistic_regression(sentiment_model, test_data):
	"""Making predictions with logistic regression"""

	top_pos_reviews = find_top_positive_reviews(test_data)
	print "\nQ3: Products in the 20 most positive reviews: \n\t%s" % ('\n --> '.join(top_pos_reviews))

	top_neg_reviews = find_top_negative_reviews(test_data)
	print "\nQ4: Products in the 20 most negative reviews: \n\t%s" % ('\n --> '.join(top_neg_reviews))

	accuracy_model = cl_utils.get_model_classification_accuracy(sentiment_model,test_data,test_data['sentiment'])
	print "\nQ5: Accuracy of the sentiment_model (test_data) is: %s" % (round(accuracy_model,2))

	return accuracy_model

def quiz1_learn_classifier_fewer_words(train_data, test_data):
	"""Learn another classifier with fewer words"""
	significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves',
		  'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed',
		  'work', 'product', 'money', 'would', 'return']

	train_data['word_count_subset'] = train_data['word_count'].dict_trim_by_keys(significant_words, exclude=False)
	test_data['word_count_subset'] = test_data['word_count'].dict_trim_by_keys(significant_words, exclude=False)

	# print train_data[0]['word_count']
	# print train_data[0]['word_count_subset']

	simple_model = gp.create_logistic_classifier_model(train_data,
													   target = 'sentiment',
													   features=['word_count_subset'])
	simple_model_coeff = simple_model.coefficients

	positive_coeff = (simple_model_coeff['value'] > 0).sum() - 1 #(intercept)
	print "\nQ7: How many of the 20 coefficients are positive: %s" % positive_coeff

	return simple_model

def main():
	try:
		print "\n**********************************"
		print "*       Predicting sentiment     *"
		print "**********************************\n"

		products = gp.load_data('../../data_sets/amazon_baby.gl/')

		#Remove punction - built-in python string
		review_without_punctuation = products['review'].apply(gp.remove_punctuation)
		products['word_count'] = gp.graphlab.text_analytics.count_words(review_without_punctuation)

		# Ignore neutral-ratings == 3
		products = products[products['rating'] != 3]

		# Create Sentiment column
		# for every rating in products-data set. Rating > 3 (positive +1) otherwise (negative -1)
		products['sentiment'] = products['rating'].apply(lambda x:+1 if x > 3 else -1)

		# 1) Train a Logistic-Classifier model
		train_data,test_data = products.random_split(.8,seed=1)
		sentiment_model = gp.create_logistic_classifier_model(train_data,
															  target = 'sentiment',
															  features=['word_count'])
		weights = sentiment_model.coefficients
		num_positive_weights = len(weights[weights['value'] >= 0])
		num_negative_weights = len(weights[weights['value'] < 0])

		print "\nQ1: How many weights are greater >= 0 is: %s" % (num_positive_weights)

		my_predictions = predict_scores_simple_data(sentiment_model,test_data)
		print "\nQ2: Lowest probability of being classified House: %s" % (np_utils.np.argmin(my_predictions)+1)

		test_data["probability"] = sentiment_model.predict(test_data,output_type='probability')

		# Compare accuracy in TEST data
		accuracy_sent_test = quiz1_make_predictions_logistic_regression(sentiment_model,test_data)
		simple_model = quiz1_learn_classifier_fewer_words(train_data,test_data)
		accuracy_simp_test = cl_utils.get_model_classification_accuracy(simple_model,test_data,test_data['sentiment'])

		# Compare accuracy in TRAINING data
		accuracy_sent_train = cl_utils.get_model_classification_accuracy(sentiment_model,train_data,train_data['sentiment'])
		accuracy_simp_train = cl_utils.get_model_classification_accuracy(simple_model,train_data,train_data['sentiment'])

		print "\nQ9: Accuracy on TRAINING data  sentiment:%s vs simple:%s" % (accuracy_sent_train, accuracy_simp_train)
		print "\nQ10: Accuracy on TEST data  sentiment:%s vs simple:%s" % (accuracy_sent_test, accuracy_simp_test)

		accuracy_majority = cl_utils.get_majority_class_accuracy(sentiment_model, test_data)
		print "\nQ11: Accuracy of the majority class classifier model: %s" % (accuracy_majority)

	except Exception as details:
		print "Error >> %s" % details
		traceback.print_exc()

if __name__ == "__main__":
	main()