from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gzip
import os
import sys
import time

import numpy as np
import tensorflow as tf

'''
value = np.random.random((5,5))
print ( value )
labels = [ 2, 1, 2, 2 ,3 ]
argmax1 = np.argmax(value, 1)
print(argmax1)
total = len(value)
print ( argmax1 == labels )
result = np.sum(argmax1 == labels)
print(result)
print(100 - ( 100.0 * result / total))
'''

SEED = 66478  # Set to None for random seed.
PIXEL_DEPTH = 255
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10
BATCH_SIZE = 64

def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read( IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    return data

train_data = extract_data('./data/train-images-idx3-ubyte.gz', 60000)

conv1_weights = tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS , 32 ], stddev=0.1, seed=SEED, dtype=tf.float32), name='conv1_weight')
conv1_biases = tf.Variable(tf.zeros([32], dtype=tf.float32), name='conv1_biases')

conv2_weights = tf.Variable(tf.truncated_normal([5, 5, 32 , 64 ], stddev=0.1, seed=SEED, dtype=tf.float32), name='conv2_weights')
conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32), name='conv2_biases')

fc1_weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512], stddev=0.1, seed=SEED, dtype=tf.float32), name='fc1_weights')
fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32), name='fc1_biases')

fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS], stddev=0.1, seed=SEED, dtype=tf.float32), name='fc2_weights')
fc2_biases = tf.Variable(tf.constant(0.1 , shape=[NUM_LABELS], dtype=tf.float32), name='fc2_biases')


def model(data, train=False, name='model'):
    with tf.name_scope(name):
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        conv = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv = tf.nn.conv2d(pool, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)

        return tf.matmul(hidden, fc2_weights) + fc2_biases



train_data_node = tf.placeholder(tf.float32 , shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), name='train_data_node')

# Training computation: logits + cross-entropy loss.
logits = model(train_data_node, True, 'train')

sess = tf.InteractiveSession()

merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("/tmp/tensorflow-test2", sess.graph)

tf.global_variables_initializer().run()

'''
print ( sess.run(conv1_weights) )
print ( sess.run(conv1_biases) )
print ( sess.run(conv2_weights) )
print ( sess.run(conv2_biases) )
print ( sess.run(fc1_weights) )
print ( sess.run(fc1_biases) )
print ( sess.run(fc2_weights) )
print ( sess.run(fc2_biases) )
'''

batch_data = train_data[0:BATCH_SIZE, ...]

result = sess.run(logits, feed_dict={train_data_node:batch_data})


print ( sess.run(conv1_weights) )
print ( sess.run(conv1_biases) )
print ( sess.run(conv2_weights) )
print ( sess.run(conv2_biases) )
print ( sess.run(fc1_weights) )
print ( sess.run(fc1_biases) )
print ( sess.run(fc2_weights) )
print ( sess.run(fc2_biases) )

#print ( result)

sess.close()

