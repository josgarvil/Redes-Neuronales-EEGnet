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
def_epochs = 120
def_dropout_rate = 0.4
def_kernLength = 64
def_lr = 0.001
def_early_stopping_patience = 10
def_test_size = 0.3
def_trainable_layers = 15

# !pip install tensorflow-determinism

# Set a seed for reproducibility
seed_value = 0

import tensorflow as tf
import numpy as np
import random
import os
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

sns.set()

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
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout, LeakyReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

# One-hot encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split



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
CNN
'''


def scheduler(epoch, lr):
    # if epoch == 1:
    #     return def_lr
    # elif epoch > 1 and epoch < 10:
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


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

    '''
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


    return Model(inputs=input1, outputs=softmax)


'''
Plot decision regions
'''


def plot_decision_regions(X, y, model, points=200):
    markers = ('o', '^')
    colors = ('red', 'blue')
    cmap = ListedColormap(colors)

    activations = model.predict(X)
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=5000, random_state=42)
    activations_2d = tsne.fit_transform(activations)

    print(X.shape)
    print(y.shape)
    print(activations_2d.shape)
    x_2 = tsne.fit_transform(X.reshape((56,28*500)))
    s = X.reshape((56,28*500))
    print(s.shape)
    print(x_2.shape)
    # x1_min, x1_max = activations_2d[:, 0].min() - 1, activations_2d[:, 0].max() + 1
    # x2_min, x2_max = activations_2d[:, 1].min() - 1, activations_2d[:, 1].max() + 1
    x1_min, x1_max = x_2[:, 0].min() - 1, x_2[:, 0].max() + 1
    x2_min, x2_max = x_2[:, 1].min() - 1, x_2[:, 1].max() + 1

    resolution = max(x1_max - x1_min, x2_max - x2_min) / float(points)

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    input = np.array([xx1.ravel(), xx2.ravel()]).T

    print(input.shape)
    Z = np.empty(0)
    for i in range(input.shape[0]):
        val = 0#model.predict(np.array(input[i]))
        if val < 0.5:
            val = 0
        if val >= 0.5:
            val = 1
        Z = np.append(Z, val)

    Z = Z.reshape(xx1.shape)

    print(Z.shape)

    plt.pcolormesh(xx1, xx2, Z, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    #  plot all samples

    classes = ['False', 'True']
    # a = activations_2d
    a = x_2
    l = y[:, 1]
    for idx, cl in enumerate(np.unique(l)):
        plt.scatter(x=a[l == cl, 0], y=a[l == cl, 1], alpha=1.0, c=colors[idx], edgecolors='black', marker=markers[idx], s=80, label=classes[idx])

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(loc='upper left')
    plt.show()


'''
Training ALL
'''

# Cross-validation (leave-one-out)

ypredict = []
metrics = []
acc2 = []

for i in range(int(length_data/trial_epochs)):
    print("Validación con trial:", i+1, "/7")

    # Divide data in training and validation (one trial for test data)
    x_train=copy.deepcopy(training_static_data)
    x_train=np.delete(x_train, np.s_[i*trial_epochs:i*trial_epochs+trial_epochs], axis=0)
    x_val=training_static_data[i*trial_epochs:i*trial_epochs+trial_epochs,:,:]

    y_train=copy.deepcopy(labels_static_data_original.transpose())
    y_train=np.delete(y_train, np.s_[i*trial_epochs:i*trial_epochs+trial_epochs], axis=0)
    y_val=labels_static_data_original.transpose()[i*trial_epochs:i*trial_epochs+trial_epochs]

    # One-hot encoder, transfor labels from 0 1 to [1 0] and [0 1]
    y_train = oh.fit_transform(y_train).transpose()
    y_val = oh.fit_transform(y_val).transpose()

    # configure the EEGNet-8,2,16 model with kernel length of 32 samples (other
    # model configurations may do better, but this is a good starting point)
    train_model = EEGNet(nb_classes=number_classes, Chans=training_static_data.shape[1],
                         Samples=training_static_data.shape[2],
                         dropoutRate=def_dropout_rate, kernLength=def_kernLength, F1=8, D=2, F2=16,
                         dropoutType='Dropout')  # clone_model(new_model)

    # compile the model and set the optimizers
    optimizer = Adam(learning_rate=def_lr)
    train_model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy', f1_metric])

    early_stopping = EarlyStopping(monitor='loss', verbose=0, baseline=None, restore_best_weights=False,
                                   patience=def_early_stopping_patience) #, start_from_epoch=30)

    #  tb = TensorBoard(log_dir="./logs", update_freq=1)
    lr_scheduler = LearningRateScheduler(scheduler,verbose=0)

    fittedModel = train_model.fit(x_train, y_train.transpose(), batch_size=def_batch_size, epochs=def_epochs,
                              verbose=2, validation_data=(x_val, y_val.transpose()),
                              callbacks=[early_stopping, lr_scheduler], shuffle=True)

    history = fittedModel.history

    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    f1 = history['f1_metric']
    val_f1 = history['val_f1_metric']

    metrics.append(val_loss)
    metrics.append(val_acc)
    metrics.append(val_f1)

    ypredict.append(train_model.predict(x_val))

    acc2.append(per_class_accuracy(train_model.predict(x_val), y_val.transpose()))

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, '-', label='Training Acc')
    plt.plot(epochs, val_acc, ':', label='Validation Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.show()

    plt.plot(epochs, loss, '-', label='Training Loss')
    plt.plot(epochs, val_loss, ':', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')
    plt.show()


    #   plot_decision_regions(x_val, y_val.transpose(), train_model, points=200)

# Save cross-validation metrics
metric_value = metrics
savemat('metrics_results_deep_learning_static.mat', {'Loss': metric_value[::3], 'Accuracy': metric_value[1::3], 'Accuracy2': acc2, 'F1score': metric_value[2::3]},  oned_as='column')
