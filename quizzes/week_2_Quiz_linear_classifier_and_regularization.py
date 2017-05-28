__author__ = 'nadyaK'
__date__ = '05/21/2017'

import ml_graphlab_utils as gp
import ml_classification_utils as cl_utils
import ml_numpy_utils as np_utils
import matplotlib.pyplot as plt
import traceback

def quiz2_theory():
	x0 = [1,1,1,1] #intercept
	x1,x2 = [2,0,3,4],[1,2,3,1]
	y_w5 = [1,-1,-1,1]
	features_w5 = np_utils.np.array([x0,x1,x2])
	weights_w5 = np_utils.np.array([0,1,-2])

	y_quiz = [1,-1,1,1]
	features_quiz = np_utils.np.array([x0,[2.5,0.3,2.8,0.5]])
	weights_quiz = np_utils.np.array([0,1])

	for features,weights,y in [[features_w5,weights_w5,y_w5],[features_quiz,weights_quiz,y_quiz]]:
		probabilities = np_utils.compute_prob_per_features(features,weights)
		print "\nprobabilities: %s" % probabilities
		derivative = np_utils.compute_derivative_for_wi(features[1],y,probabilities)
		print "\nderivative: %s" % round(derivative,2)

#==================================================================
#                 Quiz-2: Logistic Regression
#==================================================================
def quiz2_implementing_logistic_regression(products, important_words, lg_class):
	print "\n**************************************"
	print "*  Implementing Logistic Regression  *"
	print "**************************************\n"

	# set to 1 if the count of the word perfect >=1
	products['contains_perfec'] = products['perfect'] >= 1
	print "\nQ1: # of reviews containing word perfect is: %s" % products['contains_perfec'].sum()

	print "\nTransforming data to numpy arrays ...."
	feature_matrix,sentiment = np_utils.get_numpy_data(products,important_words,'sentiment')
	n_features,n_weights = feature_matrix.shape
	print "\nQ2: # of features in the feature_matrix is: %s" % n_weights

	#*******************
	# Logistic model
	#*******************
	print "\nCreating logistic model ...."
	coefficients = predict_coefficients_logistic_model(feature_matrix,sentiment, lg_class)
	print "\nQ4: As each iteration of gradient ascent passes the log- likelihood: increases"

	predictions_yi,correctly_classified = compute_correct_score_predictions(feature_matrix,coefficients)
	print "\nQ5: reviews were predicted to have positive sentiment is: %s" % correctly_classified

	accuracy = compute_accuracy_of_the_model(predictions_yi,products)
	print "\nQ6: accuracy of the model on predictions %s" % round(accuracy,2)

	word_coefficient_tuples = get_word_coeff_tuples(important_words,coefficients)

	top_words = map(lambda x:x[0],word_coefficient_tuples[:10])
	select = list({'love','easy','great','perfect','cheap'} - set(top_words))
	print "\nQ7: not present in the top 10 most positive words: %s" % select

	least_words = map(lambda x:x[0],word_coefficient_tuples[-10:])
	select_least = list({'need','work','disappointed','even'} - set(least_words))
	print "\nQ8: not present in the top 10 most negative words: %s" % select_least

def predict_coefficients_logistic_model(feature_matrix, sentiment, lg_class):
	n_features,n_weights = feature_matrix.shape
	initial_coefficients = np_utils.np.zeros(n_weights)
	#Predict coefficients
	coefficients = lg_class.logistic_regression(feature_matrix,sentiment,initial_coefficients,step_size=1e-7,max_iter=301,
											check_likelihood=False) #True)
	return coefficients

def compute_correct_score_predictions(feature_matrix, coefficients):
	# Compute the scores as a dot product between feature_matrix and coefficients.
	scores = np_utils.np.dot(feature_matrix,coefficients)
	predictions_yi = np_utils.np.array(map(lambda x:-1 if x <= 0 else 1,scores))
	correctly_classified = (predictions_yi == 1).sum()
	return predictions_yi, correctly_classified

def compute_accuracy_of_the_model(predictions_yi,products):
	num_correct = (predictions_yi == products['sentiment']).sum()
	num_mistakes = len(products) - num_correct
	accuracy = np_utils.compute_accuracy(num_correct,float(len(products)))
	print "\t\t-----------------------------------------------------"
	print '\t\t# Reviews   correctly classified =',num_correct
	print '\t\t# Reviews incorrectly classified =',num_mistakes
	print '\t\t# Reviews total                  =',len(products)
	print "\t\t-----------------------------------------------------"
	return accuracy

def get_word_coeff_tuples(important_words, coefficients):
	"""touple (word, coefficient_value) e.g ('great', 0.066..), ('love', 0.065...).
	Sort all the (word, coefficient_value) tuples by coefficient_value in descending order."""
	coefficients = list(coefficients[1:]) # exclude intercept
	word_coefficient_tuples = [(word,coefficient) for word,coefficient in zip(important_words,coefficients)]
	word_coefficient_tuples = sorted(word_coefficient_tuples,key=lambda x:x[1],reverse=True)
	return word_coefficient_tuples

#==================================================================
#                 Quiz-3: L2 regularization
#==================================================================
def quiz3_logistic_regression_l2_penalty(products,important_words, lg_class):
	print "\n**************************************"
	print "*  Logistic Regression: L2 penalty   *"
	print "**************************************\n"
	train_data,validation_data = products.random_split(.8,seed=2)
	feature_matrix_train,sentiment_train = np_utils.get_numpy_data(train_data,important_words,'sentiment')
	feature_matrix_valid,sentiment_valid = np_utils.get_numpy_data(validation_data,important_words,'sentiment')

	table = get_table_with_logistic_model(lg_class, important_words, feature_matrix_train, sentiment_train)

	coefficients = list(table["coefficients [L2=0]"][1:]) # exclude intercept
	word_coefficient_tuples = [(word, coefficient) for word, coefficient in zip(important_words, coefficients)]
	word_coefficient_tuples = sorted(word_coefficient_tuples, key=lambda x:x[1], reverse=True)
	# print word_coefficient_tuples[:5]
	positive_words = map(lambda x: x[0], word_coefficient_tuples[:5])
	negative_words = map(lambda x: x[0], word_coefficient_tuples[-5:])

	# print "Positive words: %s" % positive_words
	# print "Negative words: %s" % negative_words
	# quiz_word = ['love', 'disappointed', 'great', 'money', 'quality']

	print "\nQ1: feature_derivative_with_L2, regularize the intercept: NO"
	print "\nQ2: L2 regularization increase/decrease the log likelihood ll(w): DECREASE"
	print "\nQ3: words is not listed in either positive_words or negative_words: QUALITY"

	l2_penalty_list = [0, 4, 10, 1e2, 1e3, 1e5]
	make_coefficient_plot(table,positive_words,negative_words,l2_penalty_list)
	print "\nQ4: All coefficients consistently get smaller in size as the L2 penalty is increased -> TRUE"
	train_accuracy = create_accuracy_table(table,feature_matrix_train,sentiment_train)
	validation_accuracy = create_accuracy_table(table,feature_matrix_valid,sentiment_valid)
	print "\nComputing accuracy ....\n"
	for key in sorted(validation_accuracy.keys()):
		print "\tL2 penalty = %g" % key
		print "\ttrain accuracy = %s, validation_accuracy = %s" % (train_accuracy[key], validation_accuracy[key])
		print "\t--------------------------------------------------------------------------------"

	make_classsification_accuracy_plot(train_accuracy,validation_accuracy)

	print "\nQ6: highest accuracy on the training data: L2 penalty = 0"
	print "\nQ7: highest accuracy on the validation data: L2 penalty = 4"
	print "\nQ8: highest accuracy on the training data imply that the model is the best one: NO"

def get_table_with_logistic_model(lg_class, important_words, feature_matrix_train, sentiment_train):
	table = gp.graphlab.SFrame({'word':['(intercept)'] + important_words})

	l2_penalty_and_names = [[0,'L2=0'], [4,'L2=4'], [10,'L2=10'], [1e2,'L2=1e2'], [1e3,'L2=1e3'], [1e5,'L2=1e5']]
	for l2_penalty, name in l2_penalty_and_names:
		print "Calculating coefficients for penalty: %s ..."%name
		coefficients = lg_class.logistic_regression_with_L2(feature_matrix_train, sentiment_train,
															  initial_coefficients=np_utils.np.zeros(194),step_size=5e-6,
															  l2_penalty=l2_penalty, max_iter=501, check_likelihood=False)
		table['coefficients [%s]'%name] = coefficients
	return table

def make_coefficient_plot(table,positive_words,negative_words,l2_penalty_list):
	plt.rcParams['figure.figsize'] = 10,6

	cmap_positive = plt.get_cmap('Reds')
	cmap_negative = plt.get_cmap('Blues')

	xx = l2_penalty_list
	plt.plot(xx,[0.] * len(xx),'--',lw=1,color='k')

	table_positive_words = table.filter_by(column_name='word',values=positive_words)
	table_negative_words = table.filter_by(column_name='word',values=negative_words)
	del table_positive_words['word']
	del table_negative_words['word']

	for i in xrange(len(positive_words)):
		color = cmap_positive(0.8 * ((i + 1) / (len(positive_words) * 1.2) + 0.15))
		plt.plot(xx,table_positive_words[i:i + 1].to_numpy().flatten(),'-',label=positive_words[i],linewidth=4.0,
			color=color)

	for i in xrange(len(negative_words)):
		color = cmap_negative(0.8 * ((i + 1) / (len(negative_words) * 1.2) + 0.15))
		plt.plot(xx,table_negative_words[i:i + 1].to_numpy().flatten(),'-',label=negative_words[i],linewidth=4.0,
			color=color)

	plt.legend(loc='best',ncol=3,prop={'size':16},columnspacing=0.5)
	plt.axis([1,1e5,-1,2])
	plt.title('Coefficient path for 5 (positive & negative) words')
	plt.xlabel('L2 penalty ($\lambda$)')
	plt.ylabel('Coefficient value')
	plt.xscale('log')
	plt.rcParams.update({'font.size':18})
	plt.tight_layout()
	plt.savefig('../graphs/Coefficient_vs_L2penalty')
	plt.close()

def make_classsification_accuracy_plot(train_accuracy,validation_accuracy):
	plt.rcParams['figure.figsize'] = 10,6
	sorted_list = sorted(train_accuracy.items(),key=lambda x:x[0])
	plt.plot([p[0] for p in sorted_list],[p[1] for p in sorted_list],'bo-',linewidth=4,label='Training accuracy')
	sorted_list = sorted(validation_accuracy.items(),key=lambda x:x[0])
	plt.plot([p[0] for p in sorted_list],[p[1] for p in sorted_list],'ro-',linewidth=4,label='Validation accuracy')
	plt.xscale('symlog')
	plt.axis([0,1e3,0.78,0.786])
	plt.legend(loc='lower left')
	plt.title('Classification Accuracy vs L2 penalty')
	plt.xlabel('L2 penalty ($\lambda$)')
	plt.ylabel('Classification Accuracy')
	plt.rcParams.update({'font.size':18})
	plt.tight_layout()
	plt.savefig('../graphs/Classification_Accuracy_vs_L2penalty')
	plt.close()

def create_accuracy_table(table, feature_matrix,sentiment):
	table_accuracy = {}
	l2_penalty_and_names = [[0,'L2=0'], [4,'L2=4'], [10,'L2=10'], [1e2,'L2=1e2'], [1e3,'L2=1e3'], [1e5,'L2=1e5']]
	for l2_penalty, name in l2_penalty_and_names:
		coefficients = table['coefficients [%s]' % name]
		table_accuracy[l2_penalty] = cl_utils.get_classification_accuracy(feature_matrix,sentiment,coefficients)
	return table_accuracy

def main():
	try:
		products = gp.load_data('../../data_sets/amazon_baby_subset.gl/')

		# Sentiment: Positives (+1) & Negative (-1) reviews
		# products['sentiment'] # [1,1,1,,-1,-1,1, ......]

		important_words = gp.load_json_file('../../data_sets/important_words.json')
		# print len(important_words)

		# Remove Punctuation
		products['review_clean'] = products['review'].apply(gp.remove_punctuation)

		# Add important_words and its number of ocurrences per review
		for word in important_words:
			products[word] = products['review_clean'].apply(lambda s:s.split().count(word))
		# print products[:10]

		lg_class = cl_utils.LogisticRegression()

		# quiz2_implementing_logistic_regression(products, important_words, lg_class)

		quiz3_logistic_regression_l2_penalty(products,important_words,lg_class)

	except Exception as details:
		print (">> Exit or Errors \n%s, %s"%(details, traceback.print_exc()))

if __name__ == "__main__":
	main()