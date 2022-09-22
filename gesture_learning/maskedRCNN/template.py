#!/usr/bin/env python3

"""
Description: Deep Learning Neural Network (DLNN) approach for gesture recognition
Author: Ayusman Saha
"""
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
import argparse

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Conv1D, Dropout, MaxPooling1D, Flatten, Dense
from keras import utils

import keypoints as kp

EPOCHS = 100                # number of epochs to train the model
BATCH_SIZE = 32             # training data batch size
NORMALIZATION = 'features'  # type of data normalization

def plot_training(epochs, results):
    plt.subplots(2)

    # loss
    plt.subplot(2, 1, 1)  
    plt.plot(epochs, results['loss'], '--', label="Training")
    plt.plot(epochs, results['val_loss'], '-', label="Validation")
    plt.title('Model Performance')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()

    # accuracy
    plt.subplot(2, 1, 2)  
    plt.plot(epochs, results['acc'], '--', label="Training")
    plt.plot(epochs, results['val_acc'], '-', label="Validation")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.legend()

    plt.show()

def build_model(inputs, outputs, train_data, summary=True):
    n_timesteps, n_features, n_outputs = inputs[1], train_data[2], outputs[1]
    print("n_timesteps:",n_timesteps, "n_features:", n_features, "n_outputs:", n_outputs)
    # print(n_timesteps, n_features, n_outputs)
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(1, activation='relu'))
    model.add(Dense(n_outputs, activation='sigmoid'))

    if summary is True:
        model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

    return model

def k_fold_cross_validation(data, labels, epochs, batch_size, K=4):
    results = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
    samples = data.shape[0] // K

    for i in range(K):
        print("Processing fold {}/{}".format(i+1, K))

        # validation data and lables
        val_data = data[samples*i:samples*(i+1)]
        val_labels = labels[samples*i:samples*(i+1)]

        # training data and labels
        train_data = np.concatenate([data[:samples*i], data[samples*(i+1):]], axis=0)
        train_labels = np.concatenate([labels[:samples*i], labels[samples*(i+1):]], axis=0)
        train_data = utils.to_categorical(train_data)
        val_data = utils.to_categorical(val_data)
        # train_labels = utils.to_categorical(train_labels)

        # print(train_data.shape)
        # print(train_labels.shape)
        
        # build model
        # model = build_model(data.shape[1], labels.shape[1])
        # print(data.shape, labels.shape)

        
        # print(val_data.shape, val_labels.shape)
        
        print(train_data.shape)
        print(train_labels.shape)

        model = build_model(data.shape, labels.shape, train_data.shape)
        
        # train model
        history = model.fit(train_data, train_labels,
                            validation_data=(val_data, val_labels),
                            epochs=epochs, batch_size=batch_size)

        # record scores
        results['loss'].append(history.history['loss'])
        results['acc'].append(history.history['acc'])
        results['val_loss'].append(history.history['val_loss'])
        results['val_acc'].append(history.history['val_acc'])

        print("")

    # average results
    results['loss'] = np.mean(results['loss'], axis=0)
    results['acc'] = np.mean(results['acc'], axis=0)
    results['val_loss'] = np.mean(results['val_loss'], axis=0)
    results['val_acc'] = np.mean(results['val_acc'], axis=0)

    return results

def main(args):
    assert args.model is None or os.path.splitext(args.model)[1] == '.h5'

    # process file
    with open(args.dataset, 'r') as f:
        train, test = kp.parse(f, normalization='features', shuffle=True)

    # format training set
    train.data = kp.dataset.normalize(train.data, train.mean, train.std)
    train.labels = utils.to_categorical(train.labels)

    if args.model is None:
        # perform K-fold cross-validation
        results = k_fold_cross_validation(train.data, train.labels, EPOCHS, BATCH_SIZE)

        # visualize training
        plot_training(np.arange(EPOCHS), results)
    else:
        # build model
        model = build_model(train.data.shape[1], train.labels.shape[1], summary=True)

        # train model
        model.fit(train.data, train.labels, epochs=EPOCHS, batch_size=BATCH_SIZE)

        # save model
        model.save(args.model)

        # save data normalization parameters
        np.savez_compressed(args.model.replace('.h5', '.npz'), mean=train.mean, std=train.std)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('dataset')
    parser.add_argument('-m', '--model', metavar='model')
    args = parser.parse_args()
    main(args)
