import tensorflow
from tensorflow.keras.layers import Dense, Dropout
from keras.layers import Dense
from keras.models import Sequential

class baseline_model_1:

    def __init__(self, layers, input_shape, output_units, dropout, no_classes):
        '''
        :param layers: how many layers do you want in your neural network
        :param input_length: the dimension of the input data
        :param output_units: int the number of hidden units in the
        :param no_classes: int the number of classes within the data
        '''
        self.layers = layers
        self.input_shape = input_shape
        self.output_units = output_units
        self.dropout = dropout
        self.no_classes = no_classes

        if self.no_classes == 2:
            activation_func = 'sigmoid'
        else:
            activation_func = 'softmax'

        model = Sequential()
        model.add(Dropout(rate=self.dropout, ))

        for i in (self.layers-1):
            model.add(Dense(units=self.output_units, input_shape=self.input_shape, activation='relu'))
            model.add(Dropout(rate=self.dropout))

        model.add(Dense(units=self.no_classes, activation=activation_func))

