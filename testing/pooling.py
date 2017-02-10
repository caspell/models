import numpy as np
import tensorflow as tf
import os
import sys
import time
import gzip
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.image as image
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

#img = Image.open(imagePath)

img = image.imread(imagePath)

#plt.imshow(img)
#plt.show()

imgShape = np.shape(img)
imgNdim = np.ndim(img)

print ( np.size(img) )

height , widht , _ = imgShape



sess = tf.InteractiveSession()



sess.close()