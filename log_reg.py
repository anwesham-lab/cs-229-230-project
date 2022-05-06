# Logistic Regression for IMDB Dataset.

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import metrics
from log_reg_util import create_dictionary, transform_text
from spam import util
import matplotlib
from sklearn.linear_model import LogisticRegression

def log_reg(X_train, y_train, X_test, y_test):
	# normalize data -> features have 0 mean and unit variance

	# X_train = train["X"]

	'''
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform(X_val)
	X_test = sc.transform(X_test)
	'''

	# try using logistic regression
	clf_lr = LogisticRegression(random_state=0, class_weight = 'balanced')
	clf_lr.fit(X_train, y_train)
	# print('Logistic Regression Train Accuracy: ', clf_lr.score(X_train, y_train))
	# y_pred = clf_lr.predict_proba(X_test)[:,1]
	# predicted_probabilities = clf_lr.predict(X_test)
	predictions = clf_lr.predict(X_test)
	accuracy = metrics.accuracy_score(y_test, predictions)
	f1_score = metrics.f1_score(y_test, predictions)


	'''
	X_train_negative = X_train[y_train == 0]
	X_train_positive = X_train[y_train == 1]
	print(X_train_negative.shape)
	print(X_train_positive.shape)
	'''


def load_dataset(train_path, valid_path, test_path):
	train = pd.read_csv(train_path, header=None)
	X_train = np.asarray(train.iloc[:, 1])
	y_train = np.asarray(train.iloc[:, 0])

	valid = pd.read_csv(valid_path, header=None)
	X_valid = np.asarray(valid.iloc[:, 1])
	y_valid = np.asarray(valid.iloc[:, 0])

	test = pd.read_csv(test_path, header=None)
	X_test = np.asarray(test.iloc[:, 1])
	y_test = np.asarray(test.iloc[:, 0])

	return X_train, y_train, X_valid, y_valid, X_test, y_test

def process_labels(y_train_raw, y_valid_raw, y_test_raw):
	y_train = np.zeros(y_train_raw.shape[0])
	y_train[y_train_raw == "positive"] = 1
	
	y_valid = np.zeros(y_valid_raw.shape[0])
	y_valid[y_valid_raw == "positive"] = 1
	
	y_test = np.zeros(y_test_raw.shape[0])
	y_test[y_test_raw == "positive"] = 1
	return y_train, y_valid, y_test

def get_vectors(X_train_raw, X_valid_raw, X_test_raw):
	vectorizer = CountVectorizer(min_df = 5, max_df = 0.8)
	train_vectors = vectorizer.fit_transform(X_train_raw)
	valid_vectors = vectorizer.transform(X_valid_raw)
	test_vectors = vectorizer.transform(X_test_raw)
	return train_vectors, valid_vectors, test_vectors


def main():
	train_path = "imdb_data_train.csv"
	valid_path = "imdb_data_dev.csv"
	test_path = "imdb_data_test.csv"

	train_reviews, train_labels = util.load_sentiment_dataset(train_path)
	valid_reviews, valid_labels = util.load_sentiment_dataset(valid_path)
	test_reviews, test_labels = util.load_sentiment_dataset(test_path)

	dictionary = create_dictionary(train_reviews)
	print('Size of dictionary: ', len(dictionary))

	train_matrix = transform_text(train_reviews, dictionary)
	valid_matrix = transform_text(valid_reviews, dictionary)
	test_matrix = transform_text(test_reviews, dictionary)

	#print(train_matrix)
	#print(train_labels)

	log_reg(train_matrix, train_labels, test_matrix, test_labels)



	'''
	X_train_raw, y_train_raw, X_valid_raw, y_valid_raw, X_test_raw, y_test_raw = load_dataset(train_path, valid_path, test_path)
	X_train, X_valid, X_test = get_vectors(X_train_raw, X_valid_raw, X_test_raw)
	# print(y_train_raw)
	y_train, y_valid, y_test = process_labels(y_train_raw, y_valid_raw, y_test_raw)
	# print(y_train)
	X_train = np.asarray(X_train)
	print(type(X_train))
	print(X_train.shape)
	print(X_train)
	# print(y_train)
	'''

	# log_reg(X_train, y_train, X_valid, y_valid, X_test, y_test)


if __name__ == "__main__":
	main()