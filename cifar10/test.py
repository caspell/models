from datetime import datetime
import time

import tensorflow as tf

import cifar10

with tf.Session() as sess :


    #global_step = tf.contrib.framework.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    images, labels = cifar10.distorted_inputs()

    #logits = cifar10.inference(images)

    print ( sess.run(images) )
    print( sess.run(labels))

    #print ( logits )


