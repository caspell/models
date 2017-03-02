'''
added xavier_initializer
added dropout
'''
import numpy as np
import tensorflow as tf
import os
import sys
import time
import argparse
import gzip

from six.moves import urllib

from utils import mnist_common as cmm

from tensorflow.examples.tutorials.mnist import input_data


IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000
SEED = 66478
BATCH_SIZE = 64
NUM_EPOCHS = 10
EVAL_FREQUENCY = 100  # Number of steps between evaluations.

train_data_filename = cmm.maybe_download('train-images-idx3-ubyte.gz')
train_labels_filename = cmm.maybe_download('train-labels-idx1-ubyte.gz')
test_data_filename = cmm.maybe_download('t10k-images-idx3-ubyte.gz')
test_labels_filename = cmm.maybe_download('t10k-labels-idx1-ubyte.gz')

# Extract it into numpy arrays.
# train_data = cmm.extract_data(train_data_filename, 10)
# train_labels = cmm.extract_labels(train_labels_filename, 10)
#
# test_data = cmm.extract_data(test_data_filename, 10)
# test_labels = cmm.extract_labels(test_labels_filename, 10)


if __name__ == '__main__' :

    filename, count = 'test_file.txt.gz' , 10

    train_data = cmm.extract_data(train_data_filename, 10)

    # cmm.array_write(filename, train_data)

    test_data = cmm.array_read(filename, count)

    print (test_data)

    print ( np.shape(test_data))

    cmm.showMultiImage(test_data)
