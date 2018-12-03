from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.regularizers import l2
from keras import callbacks
import glob
#import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
layers = tf.contrib.layers
rnn = tf.contrib.rnn
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.python.ops import array_ops

# set a constant random seed for comparable results

import random 
random.seed(0)

######################## FLAGS ########################

# paths to tf binaries + splitting into validation and test set

RECORD_FILES = glob.glob('data_IEMOCAP/*')
VALIDATION_SPLIT = glob.glob('data_IEMOCAP/*_7_*')
TRAIN_SPLIT = list(set(RECORD_FILES) - set(VALIDATION_SPLIT))

# constants and flags 

Y_SHAPE = 3
N_LABELS = 6
N_FEATURES = 34
LEN_SENTENCE = 25
LEN_WORD = 60
EMBEDDING_SIZE = 300
BATCH_SIZE = 20
WORD_LSTM_REUSE = False
N_HIDDEN = 16
N_HIDDEN_2 = 6
LEARNING_RATE = 0.0001
EPOCH = int(5500/BATCH_SIZE)
STEPS = 200*EPOCH
DECAY = 30*EPOCH
DECAY_RATE = 0.5

MODEL = 'text' # can be 'multimodal' or 'text'

# run name

RUN = MODEL+'_wlen'+str(LEN_WORD)+'_slen'+str(LEN_SENTENCE)+'_batchsize'+str(BATCH_SIZE)+'_bilstm'+str(N_HIDDEN)+'/'+str(N_HIDDEN_2)+'_learning_rate'+str(LEARNING_RATE)

# path where train logs will be saved

LOGDIR = 'training_logs/'+RUN+'/'

######################## FUNCTIONS ########################

def read_from_tfrecord(filenames):
    """
    Reads and reshapes binary files from IEMOCUP data.
    """
    tfrecord_file_queue = tf.train.string_input_producer(filenames, name='queue')
    reader = tf.TFRecordReader()
    _, tfrecord_serialized = reader.read(tfrecord_file_queue)
    tfrecord_features = tf.parse_single_example(tfrecord_serialized,
                        features={
                            'audio_features'    : tf.FixedLenFeature([],tf.string),
                            'sentence_len'      : tf.FixedLenFeature([],tf.string),
                            'word_embeddings'   : tf.FixedLenFeature([],tf.string),
                            'y'                 : tf.FixedLenFeature([],tf.string),
                            'label'             : tf.FixedLenFeature([],tf.string),
                                    },  name='tf_features')
    audio_features = tf.decode_raw(tfrecord_features['audio_features'],tf.float32)
    audio_features = tf.reshape(audio_features, (N_FEATURES,LEN_WORD,LEN_SENTENCE))
    audio_features.set_shape((N_FEATURES,LEN_WORD,LEN_SENTENCE))
    
    y = tf.decode_raw(tfrecord_features['y'],tf.float32)
    y.set_shape((Y_SHAPE))
    
    label = tf.decode_raw(tfrecord_features['label'],tf.int32)
    label.set_shape((1,))
    
    sentence_len = tf.decode_raw(tfrecord_features['sentence_len'],tf.int32)
    sentence_len.set_shape((1,))
    
    return audio_features, label, sentence_len 

def myLSTM(input_size, dense_layer_sizes, output_size):
    prevSize = input_size
    model = Sequential()
    l2Reg = l2(0.01)
    for size in dense_layer_sizes:
        model.add(Dense(size, activation='relu', kernel_regularizer=l2Reg, bias_regularizer=l2Reg))
    model.add(LSTM(output_size, activation='relu', kernel_regularizer=l2Reg, recurrent_regularizer=l2Reg, bias_regularizer=l2Reg))
    model.add(Activation('softmax'))
    model.compile()
    return model

if __name__=='__main__':
    epochs = 10
    verbosity = 2
    v_split = 0.2
    callbacks = [History()]
    LSTM = myLSTM(input_size, dense_layer_sizes, output_size)
    [features, labels, sentence_len] = read_from_tf_record(TRAIN_SPLIT)
    LSTM.fit(x=None, y=None, epochs=epochs, verbose = verbosity, callbacks=callbacks, validation_split=v_split)
