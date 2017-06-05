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
def log_likelihood_metrics(lg, feature_matrix_train, sentiment_train):
	batch_size = 100
	num_passes = 10
	num_iterations = num_passes * int(len(feature_matrix_train) / batch_size)

	coefficients_sgd,log_likelihood_sgd = lg.logistic_regression_SG(feature_matrix_train,sentiment_train,
		initial_coefficients=np_utils.np.zeros(194),step_size=1e-1,batch_size=100,max_iter=num_iterations, verbose=False)

	output_file = '../graphs/Stochastic_gradient.png'
	np_plot.make_plot_log_likelihood(log_likelihood_sgd,len_data=len(feature_matrix_train),batch_size=100,
		label='stochastic gradient, step_size=1e-1',output_file=output_file)

	output_file = '../graphs/Stochastic_gradient_smallwindow.png'
	np_plot.make_plot_log_likelihood(log_likelihood_sgd,len_data=len(feature_matrix_train),batch_size=100,smoothing_window=30,
		label='stochastic gradient, step_size=1e-1',output_file=output_file)

def plot_stochastic_and_batch(lg, feature_matrix_train, sentiment_train, log_likelihood_batch):
	batch_size = 100
	num_passes = 200
	num_iterations = num_passes * int(len(feature_matrix_train) / batch_size)

	coefficients_sgd,log_likelihood_sgd = lg.logistic_regression_SG(feature_matrix_train,sentiment_train,
		initial_coefficients=np_utils.np.zeros(194),step_size=1e-1,batch_size=100,max_iter=num_iterations, verbose=False)

	np_plot.make_plot_log_likelihood(log_likelihood_sgd,len_data=len(feature_matrix_train),batch_size=100,smoothing_window=30,
		label='stochastic, step_size=1e-1')
	output_file = '../graphs/Stochastic_vs_batch.png'
	np_plot.make_plot_log_likelihood(log_likelihood_batch,len_data=len(feature_matrix_train),batch_size=len(feature_matrix_train),
		smoothing_window=1,label='batch, step_size=5e-1',output_file=output_file)

def effects_of_step_size(lg, feature_matrix_train, sentiment_train, train_data):
	batch_size = 100
	num_passes = 10
	num_iterations = num_passes * int(len(feature_matrix_train) / batch_size)

	coefficients_sgd = {}
	log_likelihood_sgd = {}
	log_space = np_utils.np.logspace(-4,2,num=7)
	for step_size in log_space:
		coefficients_sgd[step_size],log_likelihood_sgd[step_size] = lg.logistic_regression_SG(feature_matrix_train,
			sentiment_train,initial_coefficients=np_utils.np.zeros(194),step_size=step_size,batch_size=100,
			max_iter=num_iterations,verbose=False)
	for step_size in log_space[0:6]:
		np_plot.make_plot_log_likelihood(log_likelihood_sgd[step_size],len_data=len(train_data),batch_size=100,smoothing_window=30,
			label='step_size=%.1e' % step_size)
	output_file = '../graphs/Stochastic_step_sizes.png'
	np_plot.plt.savefig(output_file)
	np_plot.plt.close()

def main():
	try:
		print "\n**************************************"
		print "*          Online Learning           *"
		print "**************************************\n"

		products = gp.load_data('../../data_sets/amazon_baby_subset.gl/')
		important_words = gp.load_json_file('../../data_sets/important_words.json')

		# Remove Punctuation
		products['review_clean'] = products['review'].apply(gp.remove_punctuation)

		# Add important_words and its number of ocurrences per review
		for word in important_words:
			products[word] = products['review_clean'].apply(lambda s:s.split().count(word))
		# print products[:10]

		train_data,validation_data = products.random_split(.9,seed=1)
		feature_matrix_train,sentiment_train = np_utils.get_numpy_data(train_data,important_words,'sentiment')
		feature_matrix_valid,sentiment_valid = np_utils.get_numpy_data(validation_data,important_words,'sentiment')
		print "\nQ1: stochastic gradient ascent affect the number of features NOT: Stays the same"
		print "\nQ2: llA (w) = (1/N) * ll(w) --> only add (1/N)"
		print "\nQ3:  dli(w)/dwj is a --> scalar"
		print "\nQ4:  dli(w)/dwj (minibatch) is a: scalar"
		print "\nQ5: to have the same as the full gradient set B=N (size of train_data): %s" % len(train_data)
		print "\nQ6: logistic_regression_SG act as a standard gradient ascent when B=N (size of train_data): %s" % len(
			train_data)
		lg = cl_utils.LogisticRregStochastic()
		coefficients,log_likelihood = lg.logistic_regression_SG(feature_matrix_train,sentiment_train,
			initial_coefficients=np_utils.np.zeros(194),step_size=5e-1,batch_size=1,max_iter=10, verbose=False)
		print "\nQ7: set batch_size = 1, as each iteration passes, the average log likelihood in the batch:  Fluctuates"
		# print coefficients
		coefficients_batch,log_likelihood_batch = lg.logistic_regression_SG(feature_matrix_train,sentiment_train,
			initial_coefficients=np_utils.np.zeros(194),step_size=5e-1,batch_size=len(feature_matrix_train),
			max_iter=200, verbose=False)
		print "\nQ8: set batch_size = 47780, as each iteration passes, the average log likelihood in the batch:  Increases"
		# print coefficients_batch
		print "\nQ9: gradient updates are performed at the end of two passes  ((2*50000)/100.0) = %s" % ((2 * 50000) / 100.0)

		# log_likelihood_metrics(lg,feature_matrix_train,sentiment_train)

		plot_stochastic_and_batch(lg,feature_matrix_train,sentiment_train,log_likelihood_batch)
		print "\nQ10: passes  needed to achieve a similar log likelihood as stochastic gradient ascent: 150 passes or more"

		# effects_of_step_size(lg,feature_matrix_train,sentiment_train,train_data)
		print "\nQ11: worst step size is: 1e2"
		print "\nQ12: best step size is: 1e0"

	except Exception as details:
		print (">> Exit or Errors \n%s, %s"%(details, traceback.print_exc()))

if __name__ == "__main__":
	main()