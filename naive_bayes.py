# Naive Bayes Baseline for IMDB Dataset

import collections
# from cv2 import log
import util
import csv
import numpy as np
import argparse
import string
import sklearn
from sklearn import metrics

def load_sentiment_csv(path):
    """Load the spam dataset from a TSV file

    Args:
         csv_path: Path to TSV file containing dataset.

    Returns:
        messages: A list of string devues containing the text of each message.
        labels: The binary labels (0 or 1) for each message. A 1 indicates spam.
    """

    messages = []
    labels = []

    with open(path, 'r', newline='', encoding='utf8') as data_file:
        reader = csv.reader(data_file, delimiter=',')

        for label, message in reader:
            messages.append(message[1:-1])
            labels.append(1 if label == 'positive' else 0)

    return messages, np.array(labels)

def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing a sentiment message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    for character in string.punctuation:
        if character != "!" and character != "'":
            message = message.replace(character, ' ')
    tokens = message.split(" ")
    return [token.lower() for token in tokens if token != ""]
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    counter = collections.Counter()
    for message in messages:
        counter.update(set(get_words(message)))
    common_words = [word for word, count in counter.items() if count >= 5]
    return {word: index for index, word in enumerate(common_words)}
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message. 
    Each row in the resulting array should correspond to each message 
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***
    count_arr = np.zeros((len(messages), len(word_dictionary)))
    for index, message in enumerate(messages):
        for word in get_words(message):
            if word in word_dictionary:
                count_arr[index, word_dictionary[word]] += 1
    
    return count_arr
    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***
    
    # calculate important priors + likelihood elements
    class0 = matrix[labels == 0]
    class1 = matrix[labels == 1]
    phi = 1. * sum(labels) / len(labels)
    theta0 = (class0).sum(axis=0) + 1
    theta1 = (class1).sum(axis=0) + 1
    theta0 /= theta0.sum()
    theta1 /= theta1.sum()

    # return np.log(phi0), np.log(phi1), np.log(theta0), np.log(theta1)
    model = {}
    model['log_phi0'] = np.log(1.-phi)
    model['log_phi1'] = np.log(phi)
    model['log_theta0'] = np.log(theta0)
    model['log_theta1'] = np.log(theta1)

    return model
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    # *** START CODE HERE ***
    log_phi0 = model['log_phi0']
    log_phi1 = model['log_phi1']
    log_theta0 = model['log_theta0']
    log_theta1 = model['log_theta1']
    log_prob0 = log_phi0 + (matrix * log_theta0).sum(axis=1)
    log_prob1 = log_phi1 + (matrix * log_theta1).sum(axis=1)
    return ((log_prob1 > log_prob0).astype(int))
    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary, positive):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    if positive:
        top_indexes = np.argsort(model['log_theta0'] - model['log_theta1'])[:5]
    else:
        top_indexes = np.argsort(model['log_theta1'] - model['log_theta0'])[:5]
    index_mappings = {index: word for word, index in dictionary.items()}
    return [index_mappings[idx] for idx in top_indexes]
    # *** END CODE HERE ***


def main(train_path, dev_path, test_path, prefix):
    train_messages, train_labels = load_sentiment_csv(train_path)
    dev_messages, dev_labels = load_sentiment_csv(dev_path)
    test_messages, test_labels = load_sentiment_csv(test_path)

    dictionary = create_dictionary(train_messages)
    print('Size of dictionary: ', len(dictionary))
    util.write_json(prefix + '_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    dev_matrix = transform_text(dev_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)
    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)
    np.savetxt(prefix + '_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)
    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))
    
    f1_score = metrics.f1_score(test_labels, naive_bayes_predictions)
    print('Naive Bayes had an f1 score of {} on the testing set'.format(f1_score))

    top_5_pos_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary, True)
    print('The top 5 positive indicative words for Naive Bayes are: ', top_5_pos_words)
    util.write_json(prefix + '_positive_top_indicative_words', top_5_pos_words)

    top_5_neg_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary, False)
    print('The top 5 negative indicative words for Naive Bayes are: ', top_5_neg_words)
    util.write_json(prefix + '_negative_top_indicative_words', top_5_neg_words)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add Arguments')
    parser.add_argument('--train', default='imdb_data_train.csv', metavar='path', 
                        required=False, help='training data path')
    parser.add_argument('--dev', default='imdb_data_dev.csv', metavar='path', 
                        required=False, help='dev/validation data path')
    parser.add_argument('--test', default='imdb_data_test.csv', metavar='path', 
                        required=False, help='testing data path')
    parser.add_argument('--prefix', default='imdb', metavar='string', 
                        required=False, help='Prefix for Save Paths (Data Origin)')
    args = parser.parse_args()
    main(train_path=args.train, dev_path=args.dev, test_path=args.test, prefix=args.prefix)
