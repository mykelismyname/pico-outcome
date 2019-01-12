import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import learn
import pandas as pd
import time
import os
from datetime import datetime
from text_cnn import cnn_model
import re
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from glob import glob
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

tf.logging.set_verbosity(tf.logging.ERROR)

# defining parameters
tf.flags.DEFINE_float("sample_percentage", .1, "Percentage of data to be used for validation")
tf.flags.DEFINE_string("ebm_file", "labels_outcomes.csv", "Data source.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Misc Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer("checkpoint_every", 10, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

FLAGS = tf.flags.FLAGS

def load_span_label():
    data = pd.read_csv(FLAGS.ebm_file)
    x_text, y_text = data['Outcome'], data['Label']
    return x_text, y_text

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield (shuffled_data[start_index:end_index])

def onehot_encod_y(y):
    unique_y, unique_y_id = (list(set(y))), {}
    for i, label in enumerate(unique_y):
        unique_y_id[label] = i

    y_encod = [unique_y_id[j] for j in y if j in unique_y_id]

    # step 2, apply tensorflow onehot encoder
    with tf.Session() as sess:
        y_encod = sess.run(tf.one_hot(y_encod, depth=len(y_encod), name='one-hot-encoding'))
    return y_encod

def pre_process():
    # build a vocabularly
    x, y = load_span_label()
    max_document_length = max([len(re.split('\s+', sent.strip())) for sent in x])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length) #creates a vocabularly processor
    x_vocab = np.array(list(vocab_processor.fit_transform(x))) #processor label encods each token per sentence

    #converting y labels into one-hot vectors
    y_encod = LabelEncoder().fit_transform(y)
    y_encod = y_encod.reshape(len(y_encod), 1)
    y_encod = OneHotEncoder(sparse=False).fit_transform(y_encod)

    #shuffling the data
    np.random.seed(6)
    indices = np.random.permutation(len(y_encod))
    shuffled_x = x_vocab[indices]
    shuffled_y = y_encod[indices]

    x_train, x_val, y_train, y_val = train_test_split(shuffled_x, shuffled_y, test_size=0.2)

    del x, y, shuffled_x, shuffled_y
    return x_train, y_train, vocab_processor, x_val, y_val

def train(x_train, y_train, vocab_processor, x_val, y_val):

    #changing the default session
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
                allow_soft_placement = FLAGS.allow_soft_placement, #allows tensorflow to fall back on CPU if preferred device doesn't exist
                log_device_placement = FLAGS.log_device_placement) #TensorFlow log on which devices
        sess = tf.Session(config=session_conf)

        # overwriting the default tensorflowgraph
        with sess.as_default():
            cnn = cnn_model(sequence_length=x_train.shape[1],
                            embedding_size=FLAGS.embedding_dim,
                            vocab_size=len(vocab_processor.vocabulary_),
                            filter_sizes=list(map(int, FLAGS.filter_sizes.split(','))),
                            num_filters= FLAGS.num_filters,
                            num_classes=y_train.shape[1],
                            l2_reg_lambda=FLAGS.l2_reg_lambda
                            )

            #defining training procedure
            #minimizing the loss by first calculating the gradient values and then applying them to the variables
            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grad_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grad_and_vars, global_step=global_step)

            #create logs for the accuracies as wellas loss to track the progress of the training
            acc_summary = tf.summary.scalar('accuracy', cnn.accuracy)
            los_summary = tf.summary.scalar('loss', cnn.loss)

            #specifying a directory to store our logged summaries
            cur_time = str(datetime.now().isoformat())
            cur_dir = os.path.abspath(os.path.join('runs', cur_time))

            #train summaries
            training_summary = tf.summary.merge([acc_summary, los_summary], name='training_summary')
            training_summary_dir = tf.summary.FileWriter(os.path.join(cur_dir, 'summaries', 'train'), sess.graph)

            #test summaries
            testing_summary = tf.summary.merge([acc_summary, los_summary], name='testing_summary')
            testing_summary_dir = tf.summary.FileWriter(os.path.join(cur_dir, 'summaries', 'test'), sess.graph)

            #checkpoints, Tensorflow assumes that checkpoint directory already exists, so, make sure you let him know that you want to create one
            check_point_dir = os.path.join(cur_dir, 'checkpoints')
            checkpoint_prefix = os.path.join(check_point_dir, "model")
            if not os.path.exists(check_point_dir):
                os.makedirs(check_point_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # vocab_file = os.path.join(cur_dir, 'vocab')
            # vocab_processor.save(vocab_file)
            # vocab_text_file = os.path.abspath(os.path.join(cur_dir, 'vocab.txt'))
            # vocab_text_file.write_text(str(vocab_processor.vocabulary_._mapping))

            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                feed_dict = {cnn.input_x:x_batch, cnn.input_y:y_batch, cnn.dropout_keep_prob:FLAGS.dropout_keep_prob}
                _, step, summary, loss, accuracy = sess.run([train_op, global_step, training_summary, cnn.loss, cnn.accuracy], feed_dict)

                curr_time = datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(curr_time, step, loss, accuracy))
                training_summary_dir.add_summary(summary, step)

            def eval_step(x_batch, y_batch, writer=None):
                feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.dropout_keep_prob: FLAGS.dropout_keep_prob}
                _, step, summary, loss, accuracy = sess.run(
                    [train_op, global_step, testing_summary, cnn.loss, cnn.accuracy], feed_dict)

                curr_time = datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(curr_time, step, loss, accuracy))
                testing_summary_dir.add_summary(summary, step)

            batches = batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_checkpoints)

            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print('\nEvaluate')
                    eval_step(x_val, y_val)
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))



def main(argv=None):
    x_train, y_train, vocab_processor, x_dev, y_dev = pre_process()
    train(x_train, y_train, vocab_processor, x_dev, y_dev)

if __name__ == '__main__':
    tf.app.run()