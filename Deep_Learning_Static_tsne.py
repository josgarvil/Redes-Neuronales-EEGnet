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
import copy
import matplotlib.pyplot as plt

# TSNE
from sklearn.manifold import TSNE

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
from sklearn.metrics import accuracy_score

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

def per_class_accuracy(y_pred, y_true):

    class_labels = ['0', '1']

    guard1 = []
    guard2 = []

    for pred_idx, y_p in enumerate(y_pred):
        for class_label in class_labels:
            if y_true[pred_idx][1] == int(class_label):
                if int(class_label) == 0:
                    guard1.append(y_true[pred_idx] == np.round(y_p))
                else:
                    guard2.append(y_true[pred_idx] == np.round(y_p))

    return [np.mean(guard2), np.mean(guard1)]


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

    pre_embedding = Flatten(name='flatten')(block2)

    tsne_embedding = Dense(2, name='dense1',
                         kernel_constraint=max_norm(norm_rate),
                         kernel_initializer=glorot_uniform(seed=seed_value))(pre_embedding)
    processed_embedding = Dense(3, name='dense2',
                           kernel_constraint=max_norm(norm_rate),
                           kernel_initializer=glorot_uniform(seed=seed_value))(tsne_embedding)
    prediction = Dense(nb_classes, name='dense3', activation='softmax',
                           kernel_constraint=max_norm(norm_rate),
                           kernel_initializer=glorot_uniform(seed=seed_value))(processed_embedding)
    '''
    dense = Dense(nb_classes, name='dense',
                           kernel_constraint=max_norm(norm_rate),
                           kernel_initializer=glorot_uniform(seed=seed_value))(pre_embedding)

    softmax = Activation('softmax', name='softmax')(dense)
    '''
    return Model(inputs=input1, outputs=prediction)


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

mat_fname = pjoin('', 'vector_tasks_training.mat')  # pjoin('data_sample','.mat')
mat_contents = sio.loadmat(mat_fname)
vector_tasks_training = mat_contents['vector_tasks_training']

kernels = 1
number_classes = np.size(np.unique(labels_static_data_original))
length_data = labels_static_data_original.shape[1]
trial_epochs = vector_tasks_training.shape[1]

# We know that a trial has 56 epochs
session = 392  # a session has 392 samples (7 trials * 56 epochs)


# One-hot encoder, transfor labels from 0 1 to [1 0] and [0 1]
labels_static_data = labels_static_data_original.transpose()
oh = OneHotEncoder(sparse_output=False)
labels_static_data = oh.fit_transform(labels_static_data).transpose()

'''
Training ALL
'''


def traininig_generic_model(training_data, val_data, labels_training, labels_val, nb_classes, Chans, Samples, dropoutRate, kernLength, learning_rate, batch_size,epochs,path_model, earlystopping_patience):
    # Convert data to NHWC (trials, channels, samples, kernels) format. Set the number of kernels to 1.
    x_train = training_data.reshape(training_data.shape[0], training_data.shape[1], training_data.shape[2], kernels)
    x_val = val_data.reshape(val_data.shape[0], val_data.shape[1], val_data.shape[2], kernels)

    y_train = labels_training
    y_val = labels_val

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

    fittedModel = model.fit(x_train, y_train, batch_size = batch_size, epochs=epochs, verbose=2,
                            validation_data=(x_val, y_val), callbacks=[checkpointer,early_stopping], shuffle=True)

    history = fittedModel.history

    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    f1 = history['f1_metric']
    val_f1 = history['val_f1_metric']

    #print(np.count_nonzero(y_train == 1))
    #print(np.count_nonzero(y_train == 0))

    savemat('metrics_results_static.mat',  {'Loss': val_loss, 'Accuracy': val_acc, 'F1score': val_f1},  oned_as='column' )

    return model


# Divide data in training and validation (70% training and 30% validation)
# x_train, x_val, y_train, y_val = train_test_split(training_static_data, labels_static_data_original.transpose(), test_size=def_test_size, random_state=seed_value, stratify=labels_static_data_original.transpose())
x_train=copy.deepcopy(training_static_data)
x_train=np.delete(x_train, np.s_[0:trial_epochs], axis=0)
x_val=training_static_data[0:trial_epochs,:,:]

y_train=copy.deepcopy(labels_static_data_original.transpose())
y_train=np.delete(y_train, np.s_[0:trial_epochs], axis=0)
y_val=labels_static_data_original.transpose()[0:trial_epochs]

# One-hot encoder, transfor labels from 0 1 to [1 0] and [0 1]
y_train = oh.fit_transform(y_train).transpose()
y_val = oh.fit_transform(y_val).transpose()

net = traininig_generic_model(x_train, x_val, y_train.transpose(), y_val.transpose(), number_classes, training_static_data.shape[1],
                                training_static_data.shape[2], def_dropout_rate, def_kernLength,def_lr, def_batch_size, def_epochs,
                                'checkpoint_static.h5', def_early_stopping_patience)


labels = y_val.transpose()
#net.evaluate(x_val, labels, batch_size= def_batch_size, verbose=2)

# Extract 2D activations
activations = net.predict(x_val)

acc = accuracy_score(list(labels[:, 1]), list(np.around(activations[:, 1])))
print(acc)

acc2 = per_class_accuracy(net.predict(x_val), y_val.transpose())
print(acc2)
# net.compute_metrics(x_val, np.expand_dims(labels,axis=1), np.expand_dims(activations,axis=1), sample_weight=None)
# print(activations.shape)
# print(activations)

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=5000, random_state=42)
activations_2d = tsne.fit_transform(activations)
#print(activations_2d.embedding_)
#print(activations_2d.shape)
#print(activations_2d)
labels = y_val.transpose()
#print(labels.shape)
#print(labels)

# Plotting
plt.figure(figsize=(10, 5))
for label in np.unique(labels[:, 1]):
    indices = labels[:, 1] == label
    plt.scatter(activations_2d[indices, 0], activations_2d[indices, 1], label=f'Class {label}', alpha=0.5)

plt.legend()
plt.title('2D Activations from EEGNET')
plt.xlabel('Activation Component 1')
plt.ylabel('Activation Component 2')
plt.grid(True)
plt.show()
