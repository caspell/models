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
import math
from six.moves import urllib
import matplotlib.image as image
import matplotlib.gridspec as gridspec
from PIL import Image, ImageFilter , ImageColor
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import struct

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

def write_data(filepath, filename, num_images):
    if not tf.gfile.Exists(filepath):
        tf.gfile.MakeDirs(filepath)
    file = os.path.join(filepath, filename)
    with gzip.open(file, 'wb') as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        return data

def write_labels(filepath , filename, num_images):
    if not tf.gfile.Exists(filepath):
        tf.gfile.MakeDirs(filepath)
    file = os.path.join(filepath, filename)
    with gzip.open(file, 'wb') as bytestream:
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

def get_image( imagePath ) :
    '''
    imageDir = '/home/mhkim/data/images'
    imagePath = os.path.join(imageDir, 'number_font.png')
    '''
    imgSrc = image.imread(imagePath)
    return imgSrc


def masking ( img ) :
    mask = img.convert("L")
    return mask.point(lambda i : i < 100 and 255)

def pack2(arr, spacing=0. , power=1.):

    count, height, width, channel = np.shape(arr)

    maxWidth = 0
    maxHeight = 0

    if spacing > 0. :
        w = int(width * spacing)
        if spacing < 1.: maxWidth = width
    else :
        w = width

    for i in range(count):
        h, _, _ = np.shape(arr[i])
        maxWidth += w

        if maxHeight < h:
            maxHeight = h

    image = Image.new('RGBA', (maxWidth , maxHeight))

    buff = np.copy(image)

    def im ( i ) :
        img = arr[i]
        h , w , c = np.shape(img)
        bg = Image.new('RGBA', (w , h))
        for y in range(h) :
            for x in range(w):
                if img[y, x] > 0. :
                    bg.putpixel((x,y), ImageColor.getcolor('black', 'RGBA'))
        return bg

    for i in range(count):
        beginWidth = i * w
        endWidth =  beginWidth + width

        buff[: height, beginWidth : endWidth, : ] = im(i)

    img = Image.fromarray(buff)

    buff = img.resize(( int(maxWidth * power) , int(maxHeight * power) ))

    buff = np.array(buff)

    return buff

def pack(arr, spacing=0.):

    count, height, width, channel = np.shape(arr)

    maxWidth = 0
    maxHeight = 0

    if spacing > 0. :
        w = int(width * spacing)
        if spacing < 1.: maxWidth = width
    else :
        w = width

    for i in range(count):
        h, _, _ = np.shape(arr[i])
        maxWidth += w

        if maxHeight < h:
            maxHeight = h

    max = np.histogram(arr, density=True)[1][0]

    buff = np.full((maxHeight, maxWidth, channel) , max , dtype=np.float32)

    for i in range(count):
        beginWidth = i * w
        endWidth =  beginWidth + width
        buff[: height, beginWidth : endWidth, :] = arr[i]

    return buff

def getCanvas (height , width , channel=1 , value=-0.5 , dtype=np.float32 , image=None) :
    if image is not None :
        im = Image.open(image)
        im = im.convert("L")
        im = im.resize((width , height))
        image = np.asarray(im)
        image = np.reshape(image, (height, width , 1))
        return image
    else :
        return np.full((height, width, channel), value , dtype=dtype)

def getCanvas2 (height , width , image=None) :
    if image == None :
        im = Image.new('RGBA', (width, height), 'white')
    else :
        im = Image.open(image)
        im = im.resize((width, height))
    image = np.asarray(im)
    return image

def imageCopy2 (base, input , offset=None) :

    if type(base) == Image.Image :
        im1 = base
    elif type(base) == np.ndarray :
        im1 = Image.fromarray(base)
    else :
        im1 = Image.open(base)

    im2 = Image.fromarray(input)

    im2 = im2.convert('RGBA')

    oHeight, oWidth, oChannel = np.shape(im1)
    height, width, channel = np.shape(im2)

    maxHeight = np.maximum(oHeight , height)
    maxWidth = np.maximum(oWidth, width)

    if offset == None :
        offset = ( int(oHeight / 2) - int(height / 2) , int(oWidth / 2) - int(width / 2) )

    img1 = Image.new('RGBA', (maxWidth, maxHeight), (0,0,0,0))

    img1.paste(im1, (0,0))
    img1.paste(im2, (offset[1],offset[0]), im2)

    return np.array(img1)

def imageCopy (origin, input , offset=None) :

    if np.ndim(input) > 3 :
        return None

    oHeight, oWidth, oChannel = np.shape(origin)
    height, width, channel = np.shape(input)

    if oHeight < height or oWidth < width or oChannel < channel :
        return None

    if offset == None :
        offset = ( int(oHeight / 2) - int(height / 2) , int(oWidth / 2) - int(width / 2) , channel )

    #np.zeros_like()
    buff = np.copy(origin)

    buff[offset[0]:offset[0]+height, offset[1]:offset[1]+width, :] = input

    return buff

def packShow(arr):

    buff = pack(arr)

    plt.imshow(np.squeeze(buff, axis=np.ndim(buff) - 1))
    plt.show()

def gridView(images) :

    for i in range(len(images)) :
        imageData = images[i]

        r, c, channel = np.shape(imageData)

        rc = int(np.round(np.sqrt(channel)))

        cc = int(channel / rc)

        if channel // rc > 1 : cc += 1

        index = 0

        fig = plt.figure('figure %d' % (i + 1))

        gs = gridspec.GridSpec(rc, cc, wspace=0.0)

        ax = [plt.subplot(gs[i]) for i in (range(channel))]

        gs.update(hspace=0)

        for i in range(channel):
            _list = imageData[:, :, index:index + 1:]
            _list = np.squeeze(_list, axis=2)
            ax[index].imshow(_list)
            index += 1

    plt.show()

def get_background (image) :
    max = np.histogram(image, density=True)[1][0]
    return np.full_like(image, max)

def get_image_clean (image, value=0) :

    height , width , channel = np.shape(image)

    im = Image.fromarray(image, 'RGB')

    image = im.convert("L")

    width , height = image.size

    image = np.asarray(image)

    max = np.histogram(image, density=True)[1][0]

    for r in range(height) :
        for c in range(width):
            if image[r,c] == max : image[r,c] = value

    image = np.reshape(image, (height , width , channel))

    return image

def get_image_clean__ (image, value=0) :

    max = np.histogram(image, density=True)[1][0]

    h , w = np.shape(image)

    for r in range(h) :
        for c in range(w):
            if image[r,c] == max : image[r,c] = value

    return image


def showImageGrid(image , data , format='%d', rate=None, color='white'):

    origin = np.squeeze(image, axis=2)

    values, offsets, rect_size = data

    if np.ndim(values) == 0 :
        values = [values]

    fig, ax = plt.subplots()

    ax.imshow(origin)

    index = 0

    bbox = bbox={'facecolor':'black', 'alpha':0.5, 'pad':5}

    for y, x in offsets:

        value = values[index]

        if rate == None :
            rect = mpatches.Rectangle((x, y), rect_size, rect_size, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x+10, y + 15, np.max(value), color=color, fontsize=13 , bbox=bbox)
        elif rate=='each':
            list = [ '%d : %.2f' % (i , value[i]) for i in range(len(value)) if value[i] >= np.mean(value)]
            # list = ['%.2f' % l for l in value if l > np.mean(value)]

            txt = '\n'.join(list)
            rect = mpatches.Rectangle((x, y), rect_size, rect_size, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x + 10, y + 15, txt, color=color, fontsize=13, bbox=bbox)

        elif np.max(value) > rate :
            rect = mpatches.Rectangle((x, y), rect_size, rect_size, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x+10, y + 15, format % np.max(value), color=color, fontsize=13 , bbox=bbox)

        index += 1

    plt.draw()
    plt.show()

def parse_image ( image ) :

    img = Image.fromarray(image, 'RGBA')
    img = img.convert('RGB')
    img = img.convert('L')

    h , w = np.shape(img)

    img = np.reshape(img, (h , w , 1))

    return img


def inparse_image ( image ) :

    img = Image.fromarray(image, 'LA')

    # h , w = np.shape(img)
    #
    # img = np.reshape(img, (h , w , 1))

    return np.array(img)


def colorscale ( image ) :

    # img = Image.fromarray(image, 'RGBA')
    # img = img.convert('RGB')
    # img = img.convert('L')

    # img = image

    return np.divide(image[..., :1], [0.299, 0.587, 0.114])
    #
    # h , w = np.shape(img)
    #
    # img = np.reshape(img, (h , w , 1))
    #
    # return img


def showImage(test_data=None , show=False):
    imageBuff = []

    print(np.shape(test_data))

    # test_data = np.squeeze(test_data)

    for img in test_data :
        img = np.squeeze(img)
        for i in range(28):
            _row = []
            for j in range(28):
                _cell = img[i][j]
                if _cell < 0:
                    _cell = 0
                else:
                    _cell = 1
                _row.append(_cell)
            imageBuff.append(_row)
            print(_row)
        print('--------------------------------------------------------')

    if show :
        plt.imshow(imageBuff)
        plt.show()

def showImage2(test_data=None , show=False):
    print(np.shape(test_data))

    # test_data = np.squeeze(test_data)
    imageBuff = []

    for img in test_data :
        img = np.squeeze(img)
        for i in range(28):
            _row = []
            for j in range(28):
                _cell = img[i][j]
                _row.append(_cell)
            imageBuff.append(_row)
            print(np.array(_row))
        print('--------------------------------------------------------')

    if show :
        plt.imshow(imageBuff)
        plt.show()
