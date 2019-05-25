from keras.layers import Dense, SpatialDropout1D, Dropout, Embedding, LSTM, Conv1D, SeparableConv1D, MaxPooling1D, GlobalAveragePooling1D, Bidirectional
import tensorflow as tf
from keras.models import Sequential
import data_prep as data_prep
from keras import regularizers
from keras.initializers import Constant
from keras.regularizers import l1_l2, l1


class lstm_model:
    def __init__(self,
                 filters,
                 input_shape,
                 vocab_size,
                 embedding_dim,
                 dropout_rate,
                 no_classes,
                 kernel,
                 activation,
                 use_pretrained_embedding = False,
                 is_embedding_trainable=False,
                 embedding_matrix=None):

        self.model = Sequential()
       
        if no_classes > 2:
            activation_func = 'softmax'
        else:
            activation_func = 'sigmoid'

        with tf.device('/device:GPU:01'):
            if use_pretrained_embedding:
                self.model.add(Embedding(input_dim=vocab_size,
                                         output_dim=embedding_dim,
                                         input_length=input_shape,
                                         weights=[embedding_matrix],
                                         trainable=is_embedding_trainable))
            else:
                self.model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_shape))


            self.model.add(SpatialDropout1D(rate=dropout_rate))
            self.model.add(Bidirectional(LSTM(units=filters, kernel_initializer=kernel, activation=activation)))
            self.model.add(Dropout(rate=dropout_rate))
            self.model.add(Dense(units=no_classes, activation=activation_func))
            print(self.model.summary())




