import lstm
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam, rmsprop
import matplotlib.pyplot as plt
import os
import shutil
import data_prep
from sklearn.metrics import classification_report
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, KFold
import sys
import time
from tabulate import tabulate
sys.path.append('/users/micheala/pico-outcome-prediction/')
import data_prep

# defining parameters
tf.flags.DEFINE_float("validation_percentage", .2, "Percentage of data to be used for validation")
tf.flags.DEFINE_string("ebm_file", "/users/micheala/pico-outcome-prediction/labels_outcomes_2.csv", "Data source.")
tf.flags.DEFINE_string("glovec", "/users/micheala/pico-outcome-prediction/glove.840B.300d.txt", "pretrained_embedding")
tf.flags.DEFINE_float("dropout_rate", .2, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_integer('embedding_dim', 150, 'embedding_dim')
tf.flags.DEFINE_string("activation",  'sigmoid', "activation")    
tf.flags.DEFINE_string("kernel", "orthogonal", "kernel")
tf.flags.DEFINE_float("learning_rate", .001, "learning_rate")
tf.flags.DEFINE_integer("epochs", 200, "epochs")
tf.flags.DEFINE_integer("batch_size", 512, "batch_size") 

Flags = tf.flags.FLAGS

def load_vectorize_shuffle_data(cross_validation=False):
    load_data = pd.read_csv(Flags.ebm_file)
    x, vocab = data_prep.seq_vectorize(load_data['Outcome'])
    y, target = data_prep.one_hot_vectors(load_data['Label'])

    #shuffle data
    np.random.seed(10)
    indices = np.random.permutation(x.shape[0])
    shuffled_x = x[indices]
    shuffled_y = y[indices]
    embedding_matrix = data_prep.fetch_embeddings(Flags.glovec, vocab, Flags.embedding_dim)
    if cross_validation:
        return shuffled_x, shuffled_y, target, vocab, embedding_matrix
    else:
        x_train, x_test, y_train, y_test = train_test_split(shuffled_x, shuffled_y, test_size=Flags.validation_percentage)
        return x_train, x_test, y_train, y_test, target, vocab, embedding_matrix


def train(_xtr, _xval, _ytr, _yval,  classes, vocabularly, emb_matrix):
    kwargs = {
        'filters': 64,
        'no_classes':len(classes),
        'vocab_size': len(vocabularly)+1,
        'input_shape': _xtr.shape[1],
        'dropout_rate': Flags.dropout_rate,
        'embedding_dim': Flags.embedding_dim,
        'use_pretrained_embedding':True,
         'is_embedding_trainable':False,
         'embedding_matrix':emb_matrix
    }
    logs = os.path.join(os.path.abspath(os.curdir), 'logs_files')

    if os.path.exists(logs):
        shutil.rmtree(logs) 

    os.makedirs(logs)

    #callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10)]
    ls_model = lstm.lstm_model(**kwargs).model

    opt = Adam(lr=Flags.learning_rate)

    ls_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])

    mod = ls_model.fit(_xtr, _ytr, validation_data=(_xval, _yval), epochs=Flags.epochs, batch_size=Flags.batch_size)
 
    loss, accuracy = ls_model.evaluate(_xtr, _ytr)
    print('Training accuracy is {:.4f} and loss {:.4f}'.format(accuracy, loss))
    loss, accuracy = ls_model.evaluate(_xval, _yval)
    print('Validation accuracy is {:.4f} and loss {:.4f}'.format(accuracy, loss))

    tr_predict = ls_model.predict(_xtr)
    dev_predict = ls_model.predict(_xval)

    with open(os.path.abspath(os.path.join(os.path.curdir, 'lstm_glovec.txt')), 'w') as f:
        f.write('Evaluation Metrics \n {} \n Training Metrics \n {} '.format(
            classification_metric(_yval, dev_predict, classes),
            classification_metric(_ytr, tr_predict, classes),
        ))

    return mod, ls_model

def create_model(classes, vocabularly, kernel, emb_matrix, input_shape):
    kwargs = {'filters':64,
	      'no_classes':classes,
          'vocab_size':vocabularly,
	      'input_shape':input_shape,
	      'activation':Flags.activation,
              'kernel':kernel,
	      'dropout_rate':Flags.dropout_rate,
	      'embedding_dim':Flags.embedding_dim,
	      'use_pretrained_embedding':True,
	      'is_embedding_trainable':False,
	      'embedding_matrix':emb_matrix
	    }
    ls_model = lstm.lstm_model(**kwargs).model
    opt = Adam(lr=Flags.learning_rate)
    ls_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
    return ls_model

def train_cross_fold(x, y, classes, vocabularly, emb_matrix):
    kf = KFold(n_splits=5, shuffle=True, random_state=None)
    entries = []
    reports = []
    # callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10)]
    kwargs = {
        'filters': 64,
        'no_classes': len(classes),
        'vocab_size': len(vocabularly)+1,
        'input_shape': x.shape[1],
        'activation':Flags.activation,
        'kernel':Flags.kernel,
        'dropout_rate': Flags.dropout_rate,
        'embedding_dim': Flags.embedding_dim,
	'use_pretrained_embedding':True,
        'is_embedding_trainable':False,
        'embedding_matrix':emb_matrix
    }
    ls_model = lstm.lstm_model(**kwargs).model

    opt = Adam(lr=Flags.learning_rate)

    ls_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
	
    try:
        fold =1
        with open('lstm_per_fold.txt', 'w') as f:
            for train_index, test_index in kf.split(x):
                x_train, y_train = x[train_index], y[train_index]
                x_test, y_test = x[test_index], y[test_index]
                start_time = time.time()
                mod = ls_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=Flags.epochs, batch_size=Flags.batch_size)
                end_time = time.time()
                train_loss, train_accuracy = ls_model.evaluate(x_train, y_train)
                val_loss, val_accuracy = ls_model.evaluate(x_test, y_test)
                plotting_history(mod, fold)
                entries.append((fold,
                            '{:.4f}'.format((end_time - start_time)),
                            '{:.4f}'.format(train_accuracy),
                            '{:.4f}'.format(val_accuracy),
                            '{:.4f}'.format(train_loss),
                            '{:.4f}'.format(val_loss)))

                tr_predict = ls_model.predict(x_train)
                dev_predict = ls_model.predict(x_test)
                reports.append(('Fold:{}'.format(fold), 'Training Accuracy:{}'.format(classification_metric(y_train, tr_predict, classes)),
						    'Validation Accuracy:{}'.format(classification_metric(y_test, dev_predict, classes))))
                f.write('fold:{}\nValidation Accuracy: {}\nTraining Accuracy: {}\nValidation Metrics \n {} \n Training Metrics \n {}'.format(
                          fold, val_accuracy, train_accuracy, classification_metric(y_test, dev_predict, classes), classification_metric(y_train, tr_predict, classes)))

                fold += 1
    except Exception as e:
        print(e)

    lstm_frame = pd.DataFrame(entries, columns=['Fold', 'Time(sec)', 'Train Accuracy', 'Test Accuracy', 'Train Loss',
                                                'Validation Loss']).sort_values(by='Test Accuracy', ascending=False)
    lstm_frame = lstm_frame.round(4)

    print(tabulate(lstm_frame, headers='keys', tablefmt='psql'))

    columns = ""
    for col in lstm_frame.columns:
        columns += '{}\t|\t'.format(col)

    np.savetxt('lstm.txt', lstm_frame.values, fmt='%s', delimiter='\t', header=columns)
    with open('lstm.txt', 'a') as l:
        l.write('Mean_time:{} Mean_Train_Accuracy:{} Mean_test_score:{} Mean_Train_loss:{} Mean_Test_loss'.format(
            np.mean(lstm_frame['Time(sec)'].astype(float)),
            np.mean(lstm_frame['Train Accuracy'].astype(float)),
            np.mean(lstm_frame['Test Accuracy'].astype(float)),
            np.mean(lstm_frame['Train Loss'].astype(float)),
            np.mean(lstm_frame['Validation Loss'].astype(float))))
        l.write('\n')

    l.close()
    return ls_model

def classification_metric(y_true, y_pred, classes):
    Y = [int(np.argmax(i, axis=-1, out=None)) for i in y_pred]
    y = [int(np.argmax(i, axis=-1, out=None)) for i in y_true]
    return ls_model


def plotting_history(mod, fold):
    acc = mod.history['acc']
    val_acc = mod.history['val_acc']
    loss = mod.history['loss']
    val_loss = mod.history['val_loss']

    x = range(0, len(acc))
    fig, (accr, lss) = plt.subplots(2,1, sharey=False)
    plt.subplots_adjust(hspace=0.5)
    accr.plot(x, acc, label='Training accuracy')
    accr.plot(x, val_acc, label='Validation accuracy')
    accr.set_title('Accuracy Plot')

    lss.plot(x, loss, label='Training loss')
    lss.plot(x, val_loss, label='Validation loss')
    lss.set_title('Loss Plot')
    fig.savefig('lstm_cv_{:d}'.format(fold))
    #plt.show()

if __name__=='__main__':
    x, y, clas, voc, emb_matrix = load_vectorize_shuffle_data(cross_validation=True)
    train_cross_fold(x, y, clas, voc, emb_matrix)
