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

train_checkpoint = '/home/mhkim/data/checkpoint/mnist_cnn/'

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


class MNISTtrain :

    train_data_filename = cmm.maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = cmm.maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = cmm.maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = cmm.maybe_download('t10k-labels-idx1-ubyte.gz')

    # Extract it into numpy arrays.
    train_data = cmm.extract_data(train_data_filename, 60000)
    train_labels = cmm.extract_labels(train_labels_filename, 60000)

    test_data = cmm.extract_data(test_data_filename, 10000)
    test_labels = cmm.extract_labels(test_labels_filename, 10000)

    # Generate a validation set.
    validation_data = train_data[:VALIDATION_SIZE, ...]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_data = train_data[VALIDATION_SIZE:, ...]
    train_labels = train_labels[VALIDATION_SIZE:]
    num_epochs = NUM_EPOCHS

    train_size = train_labels.shape[0]

    def __init__(self):

        train_data_node = tf.placeholder( tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS) , name="train_data_node")
        train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,), name="train_labels_node")

        W1 = tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, 32], stddev=0.1, seed=SEED, dtype=tf.float32), name='W1')

        W2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1, seed=SEED, dtype=tf.float32), name='W2')

        b1 = tf.Variable(tf.zeros([32], dtype=tf.float32), name='bias1')
        b2 = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32), name='bias2')

        fc1_weight = tf.Variable(tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64 , 512], stddev=0.1, seed=SEED, dtype=tf.float32), name='fc1_weight')
        fc1_bias = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32), name='fc1_bias')

        fc2_weight = tf.Variable(tf.truncated_normal([512 , NUM_LABELS], stddev=0.1, seed=SEED, dtype=tf.float32), name='fc2_weight')
        fc2_bias = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS], dtype=tf.float32), name='fc2_bias')

        with tf.name_scope('model') :
            conv = tf.nn.conv2d(train_data_node, W1 , strides=[1,1,1,1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, b1), name='relu1')
            pool = tf.nn.max_pool(relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

            conv = tf.nn.conv2d(pool, W2, strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, b2))
            pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            pool_shape = pool.get_shape().as_list()

            reshape = tf.reshape(pool, [pool_shape[0] , pool_shape[1] * pool_shape[2] * pool_shape[3]])

            hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(reshape, fc1_weight) + fc1_bias), 0.5, seed=SEED)

            self.logits = tf.matmul(hidden, fc2_weight) + fc2_bias

        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits , labels=train_labels_node), name='cost')

        self.cost += 5e-4 * ( tf.nn.l2_loss(fc1_weight) + tf.nn.l2_loss(fc1_bias) + tf.nn.l2_loss(fc2_weight) + tf.nn.l2_loss(fc2_bias) )

    def get_loss (self) :
        return self.cost

    def train (self) :

        batch = tf.Variable(0, dtype=tf.float32, name='batch')

        learning_rate = tf.train.exponential_decay(0.01, batch * BATCH_SIZE, self.train_size, 0.95, staircase=True, name='decay_learning_rate')

        self.optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(self.cost, global_step=batch, name='optimizer')

        self.train_prediction = tf.nn.softmax(self.logits, name='train_prediction')



        start_time = time.time()

        with tf.Session() as sess :
            tf.global_variables_initializer().run()

            saver = tf.train.Saver()

            for step in range(int(self.num_epochs * self.train_size) // BATCH_SIZE) :
                offset = (step * BATCH_SIZE) % (self.train_size - BATCH_SIZE)
                batch_data = self.train_data[offset : (offset + BATCH_SIZE)]
                batch_labels = self.train_labels[offset:(offset + BATCH_SIZE)]

                feed_dict = { self.train_data_node : batch_data, self.train_labels_node : batch_labels }

                sess.run(self.optimizer, feed_dict = feed_dict)

                if step % EVAL_FREQUENCY == 0 :
                    l, lr, predictions = sess.run([self.cost, self.learning_rate, self.train_prediction], feed_dict=feed_dict)
                    elapsed_time = time.time() - start_time
                    start_time = time.time()

                    print('--------------------------------------------------------------------')
                    print('Step %d (epoch %.2f), %.1f ms' % (
                    step, float(step) * BATCH_SIZE / self.train_size, 1000 * elapsed_time / EVAL_FREQUENCY))
                    print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))

                    print('Step %d (epoch %.2f), %.1f ms' % (step, float(step) * BATCH_SIZE / self.train_size , 1000 * elapsed_time / EVAL_FREQUENCY ))

            saver.save(sess=sess, save_path=os.path.join(train_checkpoint, 'save.ckpt'))

            sys.stdout.flush()

            print ( "finished!" )

        '''
            test_error = error_rate(eval_in_batches(test_data, sess), test_labels)

            print('Test error: %.1f%%' % test_error)

            if FLAGS.self_test:
              print('test_error', test_error)
              assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % ( test_error,)
        '''

if __name__ == '__main__' :

    if tf.gfile.Exists(train_checkpoint):
        tf.gfile.DeleteRecursively(train_checkpoint)
    tf.gfile.MakeDirs(train_checkpoint)

    main()