# Logistic Regression Baseline for IMDB Dataset.

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from log_reg_util_small import create_dictionary, transform_text
import util
import matplotlib
from sklearn.linear_model import LogisticRegression

def log_reg(X_train, y_train, X_valid, y_valid, X_test, y_test, min_freq, step_size=0.01, max_iter=1000000, eps=1e-5):
	print("Started")
	
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)
	print("Scaled train")
	
	theta = np.zeros(X_train.shape[1])
	valid_acc = None
	for i in range(max_iter):
		preds_val = 1 / (1 + np.exp(-np.dot(X_valid, theta)))
		preds_val = [1 if x >= 0.5 else 0 for x in preds_val]
		valid_acc_new = metrics.accuracy_score(y_valid, preds_val)
		if valid_acc != None and valid_acc_new - valid_acc < 0:
			step_size *= 0.9
		elif valid_acc != None and valid_acc_new - valid_acc < eps:
			break
		valid_acc = valid_acc_new

		orig_theta = theta.copy()
		g_thetax = 1 / (1 + np.exp(-np.dot(X_train, theta)))
		update = np.dot(X_train.T, g_thetax - y_train) / X_train.shape[0]
		theta -= step_size * update
		if np.linalg.norm(theta - orig_theta) < eps:
			break

	#clf_lr = LogisticRegression(random_state=0, class_weight = 'balanced')
	#clf_lr.fit(X_train, y_train)
	print("Fit")
	# print('Logistic Regression Train Accuracy: ', clf_lr.score(X_train, y_train))
	predictions = 1 / (1 + np.exp(-np.dot(X_test, theta)))
	predictions = [1 if x >= 0.5 else 0 for x in predictions]
	#predictions = clf_lr.predict(X_test)
	print("Predict")
	accuracy = metrics.accuracy_score(y_test, predictions)
	f1_score = metrics.f1_score(y_test, predictions)
	print(f'Minimum word frequency = {min_freq}, accuracy = {accuracy}, f1_score = {f1_score}')

def main():
	train_path = "twit_senti/twitsenti_train.csv"
	train_path_few = "imdb/imdb_data_train.csv"
	valid_path = "imdb/imdb_data_dev.csv"
	test_path = "imdb/imdb_data_test.csv"

	#train_reviews, train_labels = util.load_sentiment_dataset(train_path) #ZERO
	train_reviews, train_labels = util.load_sentiment_dataset_few(train_path, train_path_few) #FEW
	valid_reviews, valid_labels = util.load_sentiment_dataset(valid_path)
	test_reviews, test_labels = util.load_sentiment_dataset(test_path)
	min_frequencies = [50, 40, 30, 20] 
	
	for freq in min_frequencies:
		dictionary = create_dictionary(train_reviews, freq)
		print('Size of dictionary: ', len(dictionary))

		train_matrix = transform_text(train_reviews, dictionary)
		valid_matrix = transform_text(valid_reviews, dictionary)
		test_matrix = transform_text(test_reviews, dictionary)
		print("Transformed")
		log_reg(train_matrix, train_labels, valid_matrix, valid_labels, test_matrix, test_labels, freq)


if __name__ == "__main__":
	main()