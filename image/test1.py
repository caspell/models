import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import matplotlib.image as mimage
import matplotlib.gridspec as mgridspec
from PIL import Image, ImageFilter , ImageColor
import matplotlib.path as mpath
import matplotlib.patches as mpatches

from utils import mnist_common as cmm

image = "/home/mhkim/사진/dream_d9e49f73ad.jpg"


img = Image.open(image)

# plt.imshow(img)
# plt.show()

print ( np.shape(img) )

h , w , c = np.shape(img)

img = np.array(img)


holder = tf.placeholder(tf.float32)


weight = tf.Variable(tf.truncated_normal([5, 5, 3, 32], stddev=0.1, seed=66478, dtype=tf.float32))
bias = tf.Variable(tf.zeros([32], dtype=tf.float32))

conv = tf.nn.conv2d(holder, weight, strides=[1,2,2,1], padding='SAME')
# relu = tf.nn.relu(conv)
# pool = tf.nn.max_pool(relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

init = tf.global_variables_initializer()

sess = tf.InteractiveSession()

sess.run(init)

# r1, r2, r3 = sess.run([ conv, relu , pool], feed_dict={ holder : [img] })

r1 = sess.run([ conv ], feed_dict={ holder : [img] })

# plt.imshow(result[0])
# plt.show()

cmm.gridView(r1[0])

sess.close()