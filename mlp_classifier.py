# MLP Classifier for IMDB Dataset.

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from log_reg_util_small import create_dictionary, transform_text
import util
import matplotlib
from sklearn.neural_network import MLPClassifier

def run(X_train, y_train, X_test, y_test, min_freq):
	print("Started")
	
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)
	print("Scaled")
	

	print(X_train)
	print(y_train)
	clf_lr = MLPClassifier(random_state=0)
	clf_lr.fit(X_train, y_train)
	print("Fit")
	# print('Logistic Regression Train Accuracy: ', clf_lr.score(X_train, y_train))
	predictions = clf_lr.predict(X_test)
	print("Predict")
	accuracy = metrics.accuracy_score(y_test, predictions)
	f1_score = metrics.f1_score(y_test, predictions)
	print(f'Minimum word frequency = {min_freq}, accuracy = {accuracy}, f1_score = {f1_score}')

def main():
	train_path = "imdb_data_train.csv"
	valid_path = "imdb_data_dev.csv"
	test_path = "imdb_data_test.csv"

	train_reviews, train_labels = util.load_sentiment_dataset(train_path)
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
		run(train_matrix, train_labels, test_matrix, test_labels, freq)


if __name__ == "__main__":
	main()