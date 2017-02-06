import numpy as np
import tensorflow as tf
import os
import sys
import time
import gzip
from PIL import Image

import matplotlib.pyplot as plt
from utils import mnist_cnn

from tensorflow.examples.tutorials.mnist import input_data

train_checkpoint = '/home/mhkim/data/checkpoint/mnist_cnn3/'

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000
SEED = 66478
BATCH_SIZE = 64
NUM_EPOCHS = 10
EVAL_FREQUENCY = 100  # Number of steps between evaluations.

imageDir = '/home/mhkim/data/images'

imagePath = os.path.join(imageDir, 'number_font.png')

img = Image.open(imagePath)


#print ( plt.imread(imagePath))

#print(img.size)

ImgData = tf.placeholder(tf.float32, [1, img.size[1], img.size[0], 3], 'img_data')

pix = img.load()

arr1 = []
for x in range(img.size[1]) :
    arr2 = []
    for y in range(img.size[0]):
        a, b, c , _ = pix[y , x]
        arr2.append((a, b, c))
    arr1.append(arr2)

sess = tf.InteractiveSession()


values = ImgData.eval({ImgData:[arr1]})

#print ( values)

pool = tf.nn.max_pool(values, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

result = pool.eval()

#print ( result)

print(result.ndim)
print(result.shape)

#print ( result[0], result[1] , result[2] , result[3] )
#res = tf.reshape(result, [result[0], result[1] * result[2] * result[3]])

#print ( res)

#print(result.reshape())

values = result[0]

plt.imshow(result[0])
plt.show()

#mnistCnn = mnist_cnn.MnistCnn()
#resultValue = mnistCnn.execute(test_data)

sess.close()