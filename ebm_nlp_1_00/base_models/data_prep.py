import tensorflow as tf
from tensorflow.contrib import learn
from keras.preprocessing import text, sequence
from keras.utils import to_categorical
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
import json

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

def one_hot_vectors(class_series, transform_back=False, label_encod_option=False):
    '''
    one-hot encods the class labels
    :return:
    '''
    classes = list(get_number_in_classes(class_series).keys())
    l_encod = LabelEncoder()
    l_encod_fit = l_encod.fit_transform(class_series)
    if label_encod_option == True:
        encoded_classes = l_encod_fit
    else:
        encoded_classes = to_categorical(l_encod_fit)
    return encoded_classes, list(l_encod.classes_)

def write_files(obj, file, writing=False):
    data_facts_dir = os.path.abspath('./data_facts')
    if not os.path.exists(data_facts_dir):
        os.makedirs(data_facts_dir)
    file_type = file.split('.')
    if writing == True:
        with open(os.path.join(data_facts_dir, file), 'w') as open_file:
            if file_type[1] == 'json':
                json.dump(obj, open_file, indent='4')
            elif file_type[1] in ['txt', 'text']:
                open_file.write(obj)
        open_file.close()

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

    vect_tf = TfidfVectorizer(**kwargs)
    vect_ct = CountVectorizer(**kwargs)
    vect_tf_data = vect_tf.fit_transform(data)
    vect_tf_data = vect_tf_data.toarray()

    vect_ct_data = vect_ct.fit_transform(data)
    vect_ct_data = vect_ct_data.toarray()
    return vect_tf_data, vect_tf.vocabulary_, vect_ct_data, vect_ct.vocabulary_

def sequence_vectorize(data):
    '''
    Preparing text to be learnt by an embedding
    :return:
    '''
    tokenizer = text.Tokenizer()
    tokenizer.fit_on_texts(data)
    seq_vect_data = tokenizer.texts_to_sequences(data)
    print(seq_vect_data)
    max_l = len(max(seq_vect_data, key=len))
    data_padded = sequence.pad_sequences(seq_vect_data, maxlen=max_l)
    return data_padded, tokenizer.word_index

def seq_vectorize(data):
    max_l = max([len(re.split('\s+', sent)) for sent in data])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_l)
    x_vocab = np.array(list(vocab_processor.fit_transform(data)))
    return x_vocab, vocab_processor.vocabulary_._mapping

#fix the randomly positioned items in words inside squared or curly braces
def fix_mispositioned_words_in_braces(x):
    pattern_x = re.compile(r'[\[\(]\s+[\w\s,;:%/._+-]+\s+[\)\]]')
    irregular_string = pattern_x.findall(x)
    if irregular_string:
        for item in irregular_string:
            regularised_string = item[0] + item[2:-2] + item[-1]
            x = x.replace(item, regularised_string)
    return x

def fetch_embeddings(pretrained_embedding_file, wordindex, EMBEDDING_DIM):
    embedding_index = {}
    file = open(pretrained_embedding_file)
    for line in file:
        values = line.split(' ')
        words = values[0]
        coef_vecs = np.asarray(values[1:], dtype='float32')[:EMBEDDING_DIM]
        embedding_index[words] = coef_vecs

    #EMBEDDING_DIM = embedding_index.get('a').shape[0]
    embedding_matrix = np.zeros((len(wordindex) + 1, EMBEDDING_DIM))

    for word, vec in wordindex.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[vec] = embedding_vector

    return embedding_matrix

def batch_iter(x, y, batch_size, shuffle=True):
    num_batches_per_epoch = int((len(x) - 1) / batch_size) + 1

    def data_generator():
        while True:
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(len(x)))
                shuffled_x = x[shuffle_indices]
                shuffled_y = y[shuffle_indices]
            else:
                shuffled_x = x
                shuffled_y = y
            for i in range(num_batches_per_epoch):
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, len(x))

                x_batch = x[start_index:end_index]
                y_batch = y[start_index:end_index]

                yield x_batch, y_batch
    return num_batches_per_epoch, data_generator()

