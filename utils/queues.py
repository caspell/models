import tensorflow as tf
import numpy as np


vals = []

queue = tf.RandomShuffleQueue(1000 , 10 , dtype=tf.float32)

enqueue_op = queue.enqueue(vals)

batch_size = tf.Variable(10)

inputs = queue.dequeue_many(batch_size)

train_op = tf.Variable(20)


init = tf.global_variables_initializer()

qr = tf.train.QueueRunner(queue, [enqueue_op] * 4)



with tf.Session() as sess :
    init.run()
    coord = tf.train.Coordinator()
    enqueue_threads = qr.create_threads(sess , coord=coord , start=True)

    for step in range(10000) :
        if coord.sohuld_stop():
            break
        sess.run(train_op)

    coord.request_stop()

    coord.join(threads)