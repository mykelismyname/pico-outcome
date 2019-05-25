"""
Created on Sun Dec 23 15:22:54 2018

@author: Micheal
"""
import numpy as np
import tensorflow as tf
import os
import sys
from tensorflow.contrib import learn
from sklearn import metrics

class cnn_model:

    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0, use_pretrained_embedding=False, embedding=None):
        # placeholder for input and output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        #generating embeddings
        with tf.device('/gpu:0'), tf.name_scope('embedding'):
            if use_pretrained_embedding:
                W = tf.constant(embedding, name='W')
            else:
                W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='W')
            embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

        # define first convolutional and maxpooling layer
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-maxpool-{}'.format(filter_size)):
                # convolutional layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                weights = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='weights')
                biases = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='biases')
                convolution_operation = tf.nn.conv2d(embedded_chars_expanded, weights, strides=[1, 1, 1, 1], padding='VALID', name='conv')

                # applying non-linearity
                activation_operation = tf.nn.relu(tf.nn.bias_add(convolution_operation, biases), name='relu')

                # maxpooling
                pooled = tf.nn.max_pool(activation_operation, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name='pool')
                pooled_outputs.append(pooled)

        # combined the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        fcl = tf.reshape(h_pool, [-1, num_filters_total])

        # dropout
        with tf.name_scope('dropout'):
            h_drop = tf.nn.dropout(fcl, self.dropout_keep_prob)

        # scores and predictions
        with tf.name_scope("output"):
            w = tf.get_variable('w', shape=[num_filters_total, num_classes], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            scores = tf.nn.xw_plus_b(h_drop, w, b, name='scores')
            predictions = tf.arg_max(scores, 1, name='predictions')
            # Calculate mean cross-entropy loss

        # loss and accuracy
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Calculate accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(predictions, tf.arg_max(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')

        #calculate the metrics
        with tf.name_scope('precision'):
            self.precision = tf.metrics.precision(tf.arg_max(self.input_y, 1), predictions)

        with tf.name_scope('recall'):
            self.recall = tf.metrics.recall(tf.arg_max(self.input_y, 1), predictions)

