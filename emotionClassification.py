import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['KERAS_BACKEND'] = 'tensorflow'
from sklearn.linear_model import LogisticRegression

import sklearn
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import tensorflow as tf
import random
random.seed(0)
tf.set_random_seed(0)
np.random.seed(0)

# set a constant random seed for comparable results
Y_SHAPE = 3
N_LABELS = 6
N_FEATURES = 34
LEN_SENTENCE = 25
LEN_WORD = 60

data = np.load("train.npz")
x = data['x']
y = data['y']
# x.shape
x = x.reshape(x.shape[0], N_FEATURES, -1)
y_hot = tf.keras.utils.to_categorical(y, num_classes=N_LABELS)
val_data = np.load("valid.npz")
val_x = val_data['x']
val_y = val_data['y']
val_x = val_x.reshape(val_x.shape[0], N_FEATURES, -1)
val_y_hot = tf.keras.utils.to_categorical(val_y, num_classes=N_LABELS)


def cnn():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(16, 3, activation='relu',
                                     input_shape=(N_FEATURES, 1500)))
    model.add(tf.keras.layers.MaxPool1D(8))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(N_LABELS))
    model.add(tf.keras.layers.Activation('softmax'))
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(x, y_hot, epochs=50, verbose=2, validation_data=(val_x, val_y_hot))
    return history

def gru():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Permute((2,1), input_shape = (N_FEATURES, LEN_SENTENCE * LEN_WORD)))
    l2Reg = tf.keras.regularizers.l2(0.01)
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=l2Reg, bias_regularizer=l2Reg)))
    model.add(tf.keras.layers.Dropout(0.9))
    model.add(tf.keras.layers.GRU(N_LABELS, activation='relu', kernel_regularizer=l2Reg, recurrent_regularizer=l2Reg, bias_regularizer=l2Reg))
    model.add(tf.keras.layers.Activation('softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x, y_hot, epochs=50, verbose=2, validation_data=(val_x, val_y_hot))
    return history
    

def lg_svm():
    model = LogisticRegression().fit(x.max(2), y)
    predict = model.predict(val_x.max(2))
    print('Logistic Regression Accuracy')
    print(sklearn.metrics.accuracy_score(predict, val_y))

    model = SVC(kernel='linear')
    model = OneVsRestClassifier(model)
    model.fit(x.max(2), y)
    predict = model.predict(val_x.max(2))
    print('SVM Accuracy')
    print(sklearn.metrics.accuracy_score(predict, val_y))

if __name__=='__main__':
    h = cnn()
    print('CNN accuracy:')
    print(h.history['val_acc'][-1])
    lg_svm()
    h = gru()
    print('GRU accuracy:')
    print(h.history['val_acc'][-1])
