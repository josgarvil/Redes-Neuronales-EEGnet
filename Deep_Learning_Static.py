# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 20:10:16 2024

@author: Jose Garcia
"""

'''
EEGNet
'''
# Parameters CNN
# Found GPU at: /device:GPU:0
def_batch_size = 32
def_epochs = 80
def_dropout_rate = 0.4
def_kernLength = 64
def_lr = 0.001
def_early_stopping_patience = 10
def_test_size = 0.3

# !pip install tensorflow-determinism

# Set a seed for reproducibility
seed_value = 0

import tensorflow as tf
import numpy as np
import random
import os

# os.environ['TF_DETERMINISTIC_OPS'] = '1'# For working on GPUs from "TensorFlow Determinism"
os.environ['PYTHONHASHSEED'] = str(seed_value)

random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


# Load MATLAB files .mat
from os.path import dirname, join as pjoin
import scipy.io as sio
import mat73
from scipy.io import savemat

# Define EEGNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout, LeakyReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import backend as K

# One-hot encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection  import train_test_split

'''
Metrics
'''


def recall_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_metric(y_true, y_pred):
    precision = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


'''
CNN
'''


def EEGNet(nb_classes, Chans = 64, Samples = 128,
             dropoutRate = 0.5, kernLength = 64, F1 = 8,
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout'):
    """ Keras Implementation of EEGNet
    http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta
    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version. For example:

        1. Depthwise Convolutions to learn spatial filters within a
        temporal convolution. The use of the depth_multiplier option maps
        exactly to the number of spatial filters learned within a temporal
        filter. This matches the setup of algorithms like FBCSP which learn
        spatial filters within each filter in a filter-bank. This also limits
        the number of free parameters to fit when compared to a fully-connected
        convolution.

        2. Separable Convolutions to learn how to optimally combine spatial
        filters across temporal bands. Separable Convolutions are Depthwise
        Convolutions followed by (1x1) Pointwise Convolutions.


    While the original paper used Dropout, we found that SpatialDropout2D
    sometimes produced slightly better results for classification of ERP
    signals. However, SpatialDropout2D significantly reduced performance
    on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
    the default Dropout in most cases.

    Assumes the input signal is sampled at 128Hz. If you want to use this model
    for any other sampling rate you will need to modify the lengths of temporal
    kernels and average pooling size in blocks 1 and 2 as needed (double the
    kernel lengths for double the sampling rate, etc). Note that we haven't
    tested the model performance with this rule so this may not work well.

    The model with default parameters gives the EEGNet-8,2 model as discussed
    in the paper. This model should do pretty well in general, although it is
	advised to do some model searching to get optimal performance on your
	particular dataset.
    We set F2 = F1 * D (number of input filters = number of output filters) for
    the SeparableConv2D layer. We haven't extensively tested other values of this
    parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D for
    overcomplete). We believe the main parameters to focus on are F1 and D.
    Inputs:

      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D.
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.
    """

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape = (Chans, Samples, 1))

    ##################################################################
    block1 = Conv2D(F1, (1, kernLength), padding='same',
                                   input_shape=(Chans, Samples, 1),
                                   use_bias=False,
                                   kernel_initializer=glorot_uniform(seed=seed_value))(input1)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                                   depth_multiplier =D,
                                   depthwise_constraint=max_norm(1.),
                                    kernel_initializer=glorot_uniform(seed=seed_value))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 5))(block1)  # can be changed
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16),
                                   use_bias =False, padding='same',
                                    kernel_initializer=glorot_uniform(seed=seed_value))(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 5))(block2)  # can be changed
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name='flatten')(block2)

    dense = Dense(nb_classes, name='dense',
                         kernel_constraint=max_norm(norm_rate),
                         kernel_initializer=glorot_uniform(seed=seed_value))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)


'''
Loading matlab files
'''

# Data should have the following format (trials, channels, samples, kernels)

# static training
mat_fname = pjoin('', 'training_static_data.mat')  # pjoin('data_sample','.mat')
try:
    mat_contents = sio.loadmat(mat_fname)
except:
    mat_contents = mat73.loadmat(mat_fname)
training_static_data = mat_contents['training_static_data']

mat_fname = pjoin('', 'labels_static_data.mat')  # pjoin('data_sample','.mat')
mat_contents = sio.loadmat(mat_fname)
labels_static_data_original = mat_contents['labels_static_data']


kernels = 1
number_classes = np.size(np.unique(labels_static_data_original))
length_data = labels_static_data_original.shape[1]

# We know that a trial has 56 epochs
session = 392  # a session has 392 samples (7 trials * 56 epochs)


# One-hot encoder, transfor labels from 0 1 to [1 0] and [0 1]
labels_static_data = labels_static_data_original.transpose()
oh = OneHotEncoder(sparse_output=False)
labels_static_data = oh.fit_transform(labels_static_data).transpose()

'''
Training ALL
'''


def traininig_generic_model(training_data, labels_training, nb_classes, Chans, Samples, dropoutRate, kernLength, learning_rate, batch_size,epochs,path_model, earlystopping_patience):
    # Convert data to NHWC (trials, channels, samples, kernels) format. Set the number of kernels to 1.
    x_train = training_data.reshape(training_data.shape[0], training_data.shape[1], training_data.shape[2], kernels)
    # x_val = val_data.reshape(val_data.shape[0], val_data.shape[1], val_data.shape[2], kernels)

    y_train = labels_training
    # y_val= labels_val

    print(np.count_nonzero(y_train == 1))
    print(np.count_nonzero(y_train == 0))

    # configure the EEGNet-8,2,16 model with kernel length of 32 samples (other
    # model configurations may do better, but this is a good starting point)
    model = EEGNet(nb_classes=nb_classes, Chans=Chans, Samples=Samples,
                dropoutRate=dropoutRate, kernLength=kernLength, F1=8, D=2, F2=16,
                dropoutType='Dropout')

    # Other 2 neural networks
    # model =ShallowConvNet(nb_classes= nb_classes, Chans =Chans, Samples = Samples, dropoutRate = dropoutRate)
    # model =DeepConvNet(nb_classes= nb_classes, Chans =Chans, Samples = Samples, dropoutRate = dropoutRate)

    # compile the model and set the optimizers
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                metrics=['accuracy', f1_metric])

    # set a valid path for your system to record model checkpoints
    checkpointer = ModelCheckpoint(filepath=path_model, verbose=1)
    #                               save_best_only=True

    # Early stopping
    early_stopping = EarlyStopping(monitor='loss', patience=earlystopping_patience)

    fittedModel = model.fit(x_train, y_train, batch_size = batch_size, epochs=epochs,
                          verbose=2, callbacks=[checkpointer,early_stopping], shuffle=True)

    history = fittedModel.history

    acc = history['accuracy']
    # val_acc = history['val_accuracy']
    loss = history['loss']
    # val_loss = history['val_loss']
    f1 = history['f1_metric']

    print(np.count_nonzero(y_train == 1))
    print(np.count_nonzero(y_train == 0))

    savemat('metrics_results_static.mat',  {'Loss': loss, 'Accuracy': acc, 'F1score': f1},  oned_as='column' )


# Divide data in training and validation (70% training and 30% validation)
# x_train, x_val, y_train, y_val = train_test_split(training_static_data, labels_static_data_original.transpose(), test_size=def_test_size, random_state=seed_value, stratify=labels_static_data_original.transpose())
x_train = training_static_data
y_train = labels_static_data_original.transpose()

# One-hot encoder, transfor labels from 0 1 to [1 0] and [0 1]
y_train = oh.fit_transform(y_train).transpose()
# y_val = oh.fit_transform(y_val).transpose()

traininig_generic_model(x_train, y_train.transpose(),number_classes, training_static_data.shape[1],
                                training_static_data.shape[2], def_dropout_rate,def_kernLength,def_lr, def_batch_size, def_epochs,
                                'checkpoint_static_pretrained.h5', def_early_stopping_patience)





