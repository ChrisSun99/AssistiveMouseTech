#!/usr/bin/env python3

"""
Description: Template for gesture recognition via machine learning
Author: Ayusman Saha (aysaha@berkeley.edu)
"""

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import models, layers, utils

import keypoints as kp

# --------------------------------------------------------------------------------------------------

# SPLIT = 0.75                # split percentage for training vs. testing data
# NORMALIZATION = 'cartesian' # type of data normalization ('cartesian' or 'polar')
K = 0                       # number of folds to process for validation
EPOCHS = 100                # number of epochs to train the model
BATCH_SIZE = 16             # training data batch size
SPLIT = 0.75                # split percentage for training vs. testing data

# NORMALIZATION = 'polar'     # type of data normalization
NORMALIZATION = 'features'  # type of data normalization ('cartesian', 'polar', or 'features')


# --------------------------------------------------------------------------------------------------
def plot(epochs, loss, acc, val_loss, val_acc):
        fig, ax = plt.subplots(2)

        # plot loss
        plt.subplot(2, 1, 1)  
        plt.plot(epochs, loss, '--b', label="Training")
        plt.plot(epochs, val_loss, '-g', label="Validation")
        plt.title('Model Performance')
        plt.ylabel('Loss')
        plt.grid()
        plt.legend()

        # plot accuracy
        plt.subplot(2, 1, 2)  
        plt.plot(epochs, acc, '--b', label="Training")
        plt.plot(epochs, val_acc, '-g', label="Validation")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.legend()

        plt.show()



def build_model(inputs, outputs, summary=False):
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=(inputs,)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    if summary is True:
        model.summary()

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

        # build model
        model = build_model(data.shape[1], labels.shape[1])

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

# NOTE: program needs keypoints.py which is located in gesture_learning/
def main(args):
    assert args.save is None or os.path.splitext(args.save)[1] == '.h5'

    # process file
    with open(args.dataset, 'r') as f:
        train, test = kp.parse(f, shuffle=True, normalization=NORMALIZATION)

    # format training set
    train.data = kp.dataset.normalize(train.data, train.mean, train.std)
    train.labels = utils.to_categorical(train.labels)

    if args.save is None:
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
        model.save(args.save)


#     train.data contains entries that are formatted as 21 (x,y) points in order. These points
#     were generated from MediaPipe and correspond to keypoints on the user's hand. When normalized,
#     this becomes a set of 20 features depending on the normalization method.

#     train.labels contains integers corresponding to different gestures. Each data entry has a
#     corresponding label arranged such that train.data[i] is categorized by train.labels[i].
#     The gesture classes for the 'fiveClass' dataset are:
#         0 - CLOSE
#         1 - OPEN
#         2 - FINGERS_ONE
#         3 - FINGERS_TWO

        # save data normalization parameters
        np.savez_compressed(args.save.replace('.h5', '.npz'), mean=train.mean, std=train.std)



# --------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('dataset')
    parser.add_argument('-s', '--save', metavar='model')
    args = parser.parse_args()
    main(args)

