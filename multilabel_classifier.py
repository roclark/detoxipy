import numpy as np
import pandas as pd
import pickle
from html import unescape
from os.path import exists
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Given the classifier outputs a binary value for all seven categories, a more
# human-readable output should be used instead. Any comment marked as 1 in the
# corresponding index to the list below falls into that category. For example,
# a value of 1 at index 0 is marked as 'Toxic'.
CLASS_REFERENCE = ['Toxic',
                   'Severe Toxic',
                   'Obscene',
                   'Thread',
                   'Insult',
                   'Identity Hate',
                   'Non-Toxic']

MODEL_NAME = 'multilabel_classifier.pkl'


def process_data(data):
    data = data.drop(labels=['id'], axis=1)
    data['non_toxic'] = 1 - data.max(axis=1)
    data['comment_text'] = data['comment_text'].apply(lambda x: unescape(x))
    train = data['comment_text']
    test = data.drop(labels=['comment_text'], axis=1)
    return train, test


def train_model(X_train, y_train):
    classifier = Pipeline([('vect', CountVectorizer(binary=True,
                                                    lowercase=True,
                                                    ngram_range=(1, 2))),
                           ('tfidf', TfidfTransformer(use_idf=False)),
                           ('clf', OneVsRestClassifier(
                                       MultinomialNB(alpha=0.01)
                                   ))
                          ])
    classifier.fit(X_train, y_train)
    return classifier


def accuracy(classifier, X_test, y_test):
    predictions = classifier.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print('Accuracy: %s%%' % round(accuracy * 100.0, 3))


def print_classes(classifications):
    for index, category in enumerate(classifications[0]):
        if category:
            print(CLASS_REFERENCE[index])


def continuous_classification(classifier):
    while True:
        try:
            comment = input('Enter comment: ')
            classifications = classifier.predict(np.array([comment]))
            print_classes(classifications)
        except KeyboardInterrupt:
            # Print an empty line to add space between the prompt and the
            # statement.
            print()
            print('User cancelled. Exiting...')
            return


def initialize_classifier():
    if exists(MODEL_NAME):
        with open(MODEL_NAME, 'rb') as f:
            classifier = pickle.load(f)
    else:
        print('No existing model found. Training new model.')
        print('This might take a while...')
        dataset = pd.read_csv('data/train.csv')
        train, test = process_data(dataset)
        X_train, X_test, y_train, y_test = train_test_split(train, test)
        classifier = train_model(X_train, y_train)
        accuracy(classifier, X_test, y_test)
        with open(MODEL_NAME, 'wb') as f:
            pickle.dump(classifier, f)
    return classifier


def main():
    classifier = initialize_classifier()
    continuous_classification(classifier)


if __name__ == '__main__':
    main()
