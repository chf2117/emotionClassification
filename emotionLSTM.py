from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.regularizers import l2

class TimeLSTM:
    def __init__(self, input_size, dense_layer_sizes, output_size):
        self.input_size = input_size
        prevSize = self.input_size
        self.model = Sequential()
        l2Reg = l2(0.01)
        for size in dense_layer_sizes:
            self.model.add(Dense(size, activation='relu', kernel_regularizer=l2Reg, bias_regularizer=l2Reg))
        self.model.add(LSTM(output_size, activation='relu', kernel_regularizer=l2Reg, recurrent_regularizer=l2Reg, bias_regularizer=l2Reg))
        model.add(Activation('softmax'))

