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
def_trainable_layers = 15

# !pip install tensorflow-determinism

# Set a seed for reproducibility
seed_value = 0

import tensorflow as tf
import numpy as np
import random
import os
import copy

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
from tensorflow.keras.models import load_model

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

    # Clone model
    train_model = load_model('checkpoint_static_pretrained.h5', custom_objects={"recall_metric": recall_metric, "precision_metric": precision_metric, "f1_metric": f1_metric  })
    # clone_model(new_model)
    for layer in train_model.layers[:-def_trainable_layers]:
        layer.trainable = False

    # Re-train clone model
    optimizer = Adam(learning_rate=def_lr)
    train_model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy', f1_metric])

    early_stopping = EarlyStopping(monitor='loss', patience=def_early_stopping_patience)

    fittedModel = train_model.fit(x_train, y_train.transpose(), batch_size=def_batch_size, epochs=def_epochs,
                              verbose =2, validation_data=(x_val, y_val.transpose()),
                              callbacks=[early_stopping], shuffle=True)

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

# Save cross-validation metrics
metric_value = metrics
savemat('metrics_results_fine_tuning_static.mat', {'Loss': metric_value[::3], 'Accuracy': metric_value[1::3], 'Accuracy2': acc2, 'F1score': metric_value[2::3]},  oned_as='column')
