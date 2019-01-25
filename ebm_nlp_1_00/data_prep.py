import tensorflow as tf
from tensorflow.python.keras.preprocessing import text, sequence
from tensorflow.python.keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder
import os
import re
import numpy as np
import pandas as pd
import matplotlib
from collections import Counter
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format='{:.2f}'.format

#file_path = 'labels_outcomes.csv'

def word_frequency(data):
    '''
    Finding out how many words, and how often they occur as well a simple plot showing us the most frequent words in the corpus
    '''
    data_text = ' '.join([o_come for o_come in data['Outcome']])
    data_text_split = text.text_to_word_sequence(data_text)
    word_vocab = Counter(data_text_split)
    print('There as many as {} words in the dataset'.format(len(word_vocab)))
    top_50_word_vocab = dict(sorted(word_vocab.items(), key=lambda x: x[1], reverse=True)[:50])
    top_50_word_vocab_df = pd.DataFrame([[k,v] for k,v in top_50_word_vocab.items()], columns=['word', 'count'])
    top_50_word_vocab_df.plot.bar()
    plt.xticks(range(len(top_50_word_vocab_df)), list(top_50_word_vocab_df['word']))
    plt.xlabel('words')
    plt.ylabel('number of occurrences')
    plt.title('Frequency of words')
    plt.figure(figsize=(14,6))
    plt.show()
    print(len(data_text_split))


def get_number_in_classes(class_series):
    '''
    Takes in all the classes for every single instance and Retrieving number unique classes and a plot of the frequency of the class appearance
    :return: a dictionary of the classes with the number of occurrences per class
    '''
    outcome_perclass = Counter(list(class_series))
    return outcome_perclass

def class_frequency(class_dict):
    outcome_perclass_df = pd.DataFrame.from_dict(class_dict, orient='index', columns=['Instances per class'])
    print('Number of classes is {}'.format(len(outcome_perclass_df.index)))
    plt.rcParams['figure.figsize'] = (7, 5)
    outcome_perclass_df.plot.bar(color='g')
    plt.show()

def one_hot_vectors(class_series):
    '''
    one-hot encods the class labels
    :return:
    '''
    classes = list(get_number_in_classes(class_series).keys())
    l_encod = LabelEncoder()
    l_encod_fit = l_encod.fit_transform(class_series)
    encoded_classes = to_categorical(l_encod_fit)
    return encoded_classes

def n_grams(data):
    '''
    vecorizing into ngrams using TFIDF
    :return: arrays of the training and texting matrices or vectorized data
    '''
    kwargs = {'ngram_range':(1,2),
              'dtype':'int32',
              'analyzer':'word',
              'decode_error':'replace',
              'strip_accents':'unicode'}

    vect = TfidfVectorizer(**kwargs)
    vect_data = vect.fit_transform(data)
    vect_data = vect_data.toarray()
    return vect_data

def sequence_vectorize(data):
    '''
    Preparing text to be learnt by an embedding
    :return:
    '''
    tok = text.Tokenize()
    tok_fit = tok.fit_on_text(data)
    seq_vect_data = tok_fit.text_to_sequences(data)

    max_len = len(max(seq_vect_data, key=len))
    data_padded = sequence.pad_sequences(seq_vect_data, maxlen=max_len)

    return data_padded
