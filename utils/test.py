'''
added xavier_initializer
added dropout
'''
import os
import sys
import time

import tensorflow as tf

from utils import mnist_common as cmm

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


def main () :

    #mnist = input_data.read_data_sets(WORK_DIRECTORY, one_hot=True)

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


    train_data_node = tf.placeholder( tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS) , name="train_data_node")
    train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,), name="train_labels_node")

    '''
        W1 = tf.get_variable("W1", shape=[784, 256], initializer=xavier_init(784, 256))
        W2 = tf.get_variable("W2", shape=[256, 256], initializer=xavier_init(256, 256))

        b1 = tf.Variable(tf.random_normal([256]), name='bias1')
        b2 = tf.Variable(tf.random_normal([256]), name='bias2')
    '''

    W1 = tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, 32], stddev=0.1, seed=SEED, dtype=tf.float32), name='W1')

    W2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1, seed=SEED, dtype=tf.float32), name='W2')

    b1 = tf.Variable(tf.zeros([32], dtype=tf.float32), name='bias1')
    b2 = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32), name='bias2')

    fc1_weight = tf.Variable(tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64 , 512], stddev=0.1, seed=SEED, dtype=tf.float32), name='fc1_weight')
    fc1_bias = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32), name='fc1_bias')

    fc2_weight = tf.Variable(tf.truncated_normal([512 , NUM_LABELS], stddev=0.1, seed=SEED, dtype=tf.float32), name='fc2_weight')
    fc2_bias = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS], dtype=tf.float32), name='fc2_bias')

    #def error_rate(predictions, labels):
    #  """Return the error rate based on dense predictions and sparse labels."""
    #  return 100.0 - ( 100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0])

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

        logits = tf.matmul(hidden, fc2_weight) + fc2_bias

    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits , labels=train_labels_node), name='cost')

    cost += 5e-4 * ( tf.nn.l2_loss(fc1_weight) + tf.nn.l2_loss(fc1_bias) + tf.nn.l2_loss(fc2_weight) + tf.nn.l2_loss(fc2_bias) )

    batch = tf.Variable(0, dtype=tf.float32, name='batch')

    learning_rate = tf.train.exponential_decay(0.01, batch * BATCH_SIZE, train_size, 0.95, staircase=True, name='decay_learning_rate')

    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cost, global_step=batch, name='optimizer')

    train_prediction = tf.nn.softmax(logits, name='train_prediction')

    start_time = time.time()

    with tf.Session() as sess :
        tf.global_variables_initializer().run()

        saver = tf.train.Saver()

        for step in range(int(num_epochs * train_size) // BATCH_SIZE) :
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = train_data[offset : (offset + BATCH_SIZE)]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]

            feed_dict = { train_data_node : batch_data, train_labels_node : batch_labels }

            sess.run(optimizer, feed_dict = feed_dict)

            if step % EVAL_FREQUENCY == 0 :
                l, lr, predictions = sess.run([cost, learning_rate, train_prediction], feed_dict=feed_dict)
                elapsed_time = time.time() - start_time
                start_time = time.time()

                print('--------------------------------------------------------------------')
                print('Step %d (epoch %.2f), %.1f ms' % (
                step, float(step) * BATCH_SIZE / train_size, 1000 * elapsed_time / EVAL_FREQUENCY))
                print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                #print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
                #print('Validation error: %.1f%%' % error_rate(eval_in_batches(validation_data, sess), validation_labels))

                print('Step %d (epoch %.2f), %.1f ms' % (step, float(step) * BATCH_SIZE / train_size , 1000 * elapsed_time / EVAL_FREQUENCY ))
                                                                              #, error_rate(eval_in_batches(validation_data, sess), validation_labels) ))

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
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--use_fp16',
        default=False,
        help='Use half floats instead of full floats if True.',
        action='store_true')
    parser.add_argument(
        '--self_test',
        default=False,
        action='store_true',
        help='True if running a self test.')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    '''