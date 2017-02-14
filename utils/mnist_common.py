from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import gzip
import os
import sys
import time
from six.moves import urllib
import matplotlib.image as image

train_checkpoint = '/home/mhkim/data/checkpoint/mnist_cnn/save.ckpt'

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = '/home/mhkim/data/mnist'

IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000
SEED = 66478
BATCH_SIZE = 64
NUM_EPOCHS = 10
EVAL_FREQUENCY = 100  # Number of steps between evaluations.
EVAL_BATCH_SIZE = 64

def maybe_download(filename):
    if not tf.gfile.Exists(WORK_DIRECTORY):
        tf.gfile.MakeDirs(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath


def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        return data

def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

def error_rate(predictions, labels):
    return 100.0 - ( 100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0])

def xavier_init(n_inputs, n_outputs, uniform=True):
    if uniform :
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else :
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

def get_image() :
    imageDir = '/home/mhkim/data/images'
    imagePath = os.path.join(imageDir, 'number_font.png')
    imgSrc = image.imread(imagePath)
    return imgSrc


def masking ( img ) :
    mask = img.convert("L")
    return mask.point(lambda i : i < 100 and 255)

def pack(arr):

    count, height, width, channel = np.shape(arr)

    maxWidth = 0
    maxHeight = 0
    for i in range(count):
        h, w, c = np.shape(arr[i])
        maxWidth += w
        if maxHeight < h:
            maxHeight = h

    buff = np.zeros((maxHeight, maxWidth, channel))

    for i in range(count):
        buff[:height, i * width:i * width + width, :] = arr[i]

    return buff

def getCanvas (height , width , channel=1) :
    return np.zeros((height , width , channel))

def imageCopy (origin, input) :

    if np.ndim(input) > 3 :
        return None

    oHeight, oWidth, oChannel = np.shape(origin)
    height, width, channel = np.shape(input)

    if oHeight < height or oWidth < height or oChannel < channel :
        return None

    offset = ( int(oHeight / 2) - int(height / 2) , int(oWidth / 2) - int(width / 2) , channel )

    #np.zeros_like()
    buff = np.copy(origin)

    buff[offset[0]:offset[0]+height, offset[1]:offset[1]+width, :] = input

    return buff

def packShow(arr):

    buff = pack(arr)

    plt.imshow(np.squeeze(buff, axis=np.ndim(buff) - 1))
    plt.show()