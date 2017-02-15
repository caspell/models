import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import mnist_common as cmm
import time
import os
import sys

class MnistCnn :

    train_checkpoint = '/home/mhkim/data/checkpoint/mnist_cnn/save.ckpt'

    IMAGE_SIZE = 28
    NUM_CHANNELS = 1
    NUM_LABELS = 10
    SEED = 66478
    PIXEL_DEPTH = 255
    VALIDATION_SIZE = 5000
    SEED = 66478
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    EVAL_FREQUENCY = 100  # Number of steps between evaluations.

    def __init__(self) :

        train_data_filename = cmm.maybe_download('train-images-idx3-ubyte.gz')
        train_labels_filename = cmm.maybe_download('train-labels-idx1-ubyte.gz')
        test_data_filename = cmm.maybe_download('t10k-images-idx3-ubyte.gz')
        test_labels_filename = cmm.maybe_download('t10k-labels-idx1-ubyte.gz')

        # Extract it into numpy arrays.
        self.train_data = cmm.extract_data(train_data_filename, 60000)
        self.train_labels = cmm.extract_labels(train_labels_filename, 60000)

        self.test_data = cmm.extract_data(test_data_filename, 10000)
        self.test_labels = cmm.extract_labels(test_labels_filename, 10000)

        # Generate a validation set.
        # validation_data = train_data[:self.VALIDATION_SIZE, ...]
        # validation_labels = train_labels[:self.VALIDATION_SIZE]


        self.W1 = tf.Variable(tf.truncated_normal([5, 5, self.NUM_CHANNELS, 32], stddev=0.1, seed=self.SEED, dtype=tf.float32), name='W1')
        self.W2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1, seed=self.SEED, dtype=tf.float32), name='W2')

        self.b1 = tf.Variable(tf.zeros([32], dtype=tf.float32), name='bias1')
        self.b2 = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32), name='bias2')

        self.fc1_weight = tf.Variable( tf.truncated_normal([self.IMAGE_SIZE // 4 * self.IMAGE_SIZE // 4 * 64, 512], stddev=0.1, seed=self.SEED, dtype=tf.float32), name='fc1_weight')
        self.fc1_bias = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32), name='fc1_bias')

        self.fc2_weight = tf.Variable(tf.truncated_normal([512, self.NUM_LABELS], stddev=0.1, seed=self.SEED, dtype=tf.float32), name='fc2_weight')
        self.fc2_bias = tf.Variable(tf.constant(0.1, shape=[self.NUM_LABELS], dtype=tf.float32), name='fc2_bias')

    def model (self, data, name='model', train=False):
        with tf.name_scope(name):
            conv = tf.nn.conv2d(data, self.W1, strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, self.b1), name='relu1')
            pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            conv = tf.nn.conv2d(pool, self.W2, strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, self.b2))
            pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            pool_shape = pool.get_shape().as_list()

            reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

            hidden = tf.nn.relu(tf.matmul(reshape, self.fc1_weight) + self.fc1_bias)

            if train:
                hidden = tf.nn.dropout(hidden, 0.5, seed=self.SEED)

            return tf.matmul(hidden, self.fc2_weight) + self.fc2_bias

    def train (self) :

        if tf.gfile.Exists(self.train_checkpoint):
            tf.gfile.DeleteRecursively(self.train_checkpoint)
        tf.gfile.MakeDirs(self.train_checkpoint)


        train_data = self.train_data[self.VALIDATION_SIZE:, ...]
        train_labels = self.train_labels[self.VALIDATION_SIZE:]

        train_size = train_labels.shape[0]

        train_data_node = tf.placeholder(tf.float32, shape=(self.BATCH_SIZE, self.IMAGE_SIZE, self.IMAGE_SIZE, self.NUM_CHANNELS),
                                         name="train_data_node")
        train_labels_node = tf.placeholder(tf.int64, shape=(self.BATCH_SIZE,), name="train_labels_node")

        logits = self.model(train_data_node , 'model', True)

        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=train_labels_node),
                              name='cost')

        regularizers = (
        tf.nn.l2_loss(self.fc1_weight) + tf.nn.l2_loss(self.fc1_bias) + tf.nn.l2_loss(self.fc2_weight) + tf.nn.l2_loss(
            self.fc2_bias))

        cost += 5e-4 * regularizers

        batch = tf.Variable(0, dtype=tf.float32, name='batch')

        learning_rate = tf.train.exponential_decay(0.01, batch * self.BATCH_SIZE, train_size, 0.95, staircase=True,
                                                   name='decay_learning_rate')

        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cost, global_step=batch, name='optimizer')

        train_prediction = tf.nn.softmax(logits, name='train_prediction')

        start_time = time.time()

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            saver = tf.train.Saver()

            for step in range(int(self.NUM_EPOCHS * train_size) // self.BATCH_SIZE):
                offset = (step * self.BATCH_SIZE) % (train_size - self.BATCH_SIZE)
                batch_data = train_data[offset: (offset + self.BATCH_SIZE)]
                batch_labels = train_labels[offset:(offset + self.BATCH_SIZE)]

                feed_dict = {train_data_node: batch_data, train_labels_node: batch_labels}

                sess.run(optimizer, feed_dict=feed_dict)

                if step % self.EVAL_FREQUENCY == 0:
                #    l, lr, predictions = sess.run([cost, learning_rate, train_prediction], feed_dict=feed_dict)
                    elapsed_time = time.time() - start_time
                    #start_time = time.time()
                    print ('Step %d , %.2f s' % (step, elapsed_time))
                #
                #     print('--------------------------------------------------------------------')
                #     print('Step %d (epoch %.2f), %.1f ms' % (
                #         step, float(step) * self.BATCH_SIZE / train_size, 1000 * elapsed_time / self.EVAL_FREQUENCY))
                #     print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                #     # print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
                #     # print('Validation error: %.1f%%' % error_rate(eval_in_batches(validation_data, sess), validation_labels))
                #
                #     print('Step %d (epoch %.2f), %.1f ms' % (
                #     step, float(step) * self.BATCH_SIZE / train_size, 1000 * elapsed_time / self.EVAL_FREQUENCY))
                #     # , error_rate(eval_in_batches(validation_data, sess), validation_labels) ))

            elapsed_time = time.time() - start_time

            print('total time : %.2f s' % elapsed_time)

            saver.save(sess=sess, save_path=os.path.join(self.train_checkpoint, 'save.ckpt'))

            sys.stdout.flush()

            print("finished!")

    def showImage(self, test_data=None , show=False):
        imageBuff = []
        for i in range(28):
            _row = []
            for j in range(28):
                _cell = test_data[0][i][j][0]
                # if _cell < 0:
                #     _cell = 0
                # else:
                #     _cell = 1
                _row.append(_cell)
            imageBuff.append(_row)
            print(_row)

        if show :
            plt.imshow(imageBuff)
            plt.show()

    def execute (self, data) :

        eval_data = tf.placeholder(tf.float32, shape=(1, self.IMAGE_SIZE, self.IMAGE_SIZE, self.NUM_CHANNELS),
                                   name='eval_data')

        logits = self.model(eval_data, 'model')

        self.eval_prediction = tf.nn.softmax(logits)

        sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, self.train_checkpoint)

        result = tf.argmax(self.eval_prediction, 1)

#        print ( self.eval_prediction.eval({eval_data: data}) )

        resultValue = result.eval({eval_data: data})[0]

        #self.showImage(data)

        sess.close()

        return resultValue

if __name__ == '__main__' :

    test_data_filename = cmm.maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = cmm.maybe_download('t10k-labels-idx1-ubyte.gz')

    test_data = cmm.extract_data(test_data_filename, 1)
    test_labels = cmm.extract_labels(test_labels_filename, 1)

    mnistCnn = MnistCnn()

    print ( '------------------------------------------------------------------' )

    resultValue = mnistCnn.execute(test_data)

    print ( resultValue )

    # mnistCnn.train()