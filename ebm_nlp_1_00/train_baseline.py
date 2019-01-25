from tensorflow.python.keras import layers
from mlp_classifier import baseline_model
import data_prep as data_prep
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
from tensorflow.python.keras.optimizers import Adam
from numba import jit
import numpy
import matplotlib.pyplot as plt
import sys
import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 100
pd.options.display.max_columns = 5

# defining parameters
tf.flags.DEFINE_float("sample_percentage", .3, "Percentage of data to be used for validation")
tf.flags.DEFINE_string("ebm_file", "labels_outcomes.csv", "Data source.")

tf.flags.DEFINE_float("dropout", .2, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate")
tf.flags.DEFINE_integer("epochs", 1000, "epochs")
tf.flags.DEFINE_integer("batch_size", 120, "batch_size")

Flags = tf.flags.FLAGS

class data_prepare:
    def __init__(self, path):
        self.path = path
        self.dataset = pd.read_csv(self.path)

def load_vectorize_split_data():
    read_file = data_prepare(Flags.ebm_file)
    data = read_file.dataset
    num_classes = len(data_prep.get_number_in_classes(data['Label']))
    #shuffle data twice
    shuffled_data = data.sample(frac=1, random_state=123).reset_index()
    shuffled_data = shuffled_data.reindex(np.random.permutation(shuffled_data.index))

    #vectorize the data
    y_values =  data_prep.one_hot_vectors(shuffled_data['Label'])
    x_values = data_prep.n_grams(shuffled_data['Outcome'])

    #split the data into training and test sets
    xtrain, xtest, ytrain, ytest = train_test_split(x_values, y_values, test_size=Flags.sample_percentage)
    return xtrain, xtest, ytrain, ytest, num_classes

def compute_loss(x):
    loss = 'binary_crossentropy'
    if x > 2:
        loss = 'categorical_crossentropy'
    return loss

def train(_xtr, _xval, _ytr, _yval, classes):
    mlpmodel = baseline_model(layers = 4, input_shape=_xtr[1:], output_units=64, dropout=Flags.dropout, no_classes=classes)
    opt = Adam(lr=Flags.learning_rate)
    mlpmodel.model.compile(optimizer=opt, loss=compute_loss(classes), metrics=['accuracy'])
    mod = mlpmodel.model.fit(_xtr, _ytr, validation_data=(_xval, _yval), batch_size=Flags.batch_size, epochs=Flags.epochs)
    loss, accuracy = mod.evaluate(_xtr, _ytr)
    print('Training accuracy is {:.4f} and loss {:.4f}'.format(accuracy, loss))
    loss, accuracy = mod.evaluate(_xval, _yval)
    print('Training accuracy is {:.4f} and loss {:.4f}'.format(accuracy, loss))


def plotting_history(mod):
    acc = mod.history['acc']
    val_acc = mod.history['val_acc']
    loss = mod.history['loss']
    val_loss = mod.history['val_loss']

    x = range(0, len(acc)+2)
    fig, (accr, lss) = plt.subplots(1,2, sharey=False)
    accr.plot(x, acc, label='Training accuracy')
    accr.plot(x, val_acc, label='Validation accuracy')

    lss.plot(x, loss, label='Training loss')
    lss.plot(x, val_loss, label='Validation loss')

    plt.show()

if __name__=='__main__':
    x_train, x_val, y_train, y_val, cls = load_vectorize_split_data()
    model = train(x_train, x_val, y_train, y_val, cls)
    plotting_history(model)





