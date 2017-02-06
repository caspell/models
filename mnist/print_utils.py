import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

def loadAndPrint ( ) :
    mnist = input_data.read_data_sets("../mnist/data/", one_hot=True)

    batch_xs, batch_ys = mnist.train.next_batch(10)
    test_xs, test_ys = mnist.test.next_batch(10)

    # print(len(batch_xs[0]))

    def printMNIST(ditis, lables):
        for n in range(len(ditis)):
            _index = 0
            print([i for i in range(len(lables[n])) if lables[n][i] > 0.])

            for i in range(28):
                _v2 = []
                for j in range(28):
                    if ditis[n][_index] > 0.:
                        # value = '%f' % ( int(batch_xs[0][_index] * 100) / 100 )
                        value = '1'
                    else:
                        value = ' '
                    _v2.append(value)
                    _index = _index + 1
                print(_v2)

    printMNIST(batch_xs, batch_ys)

#    print('{:_<150}\n'.format('_'))

#    printMNIST(test_xs, test_ys)

def printMNIST ( ditis , lables ) :
    for n in range(len(ditis)) :
        _index = 0
        print([ i for i in range(len(lables[n])) if lables[n][i] > 0. ])

        for i in range(28):
            _v2 = []
            for j in range(28) :
                if ditis[n][_index] > 0. :
                    #value = '%f' % ( int(batch_xs[0][_index] * 100) / 100 )
                    value = '1'
                else :
                    value = ' '
                _v2.append(value)
                _index = _index + 1
            print(_v2)

def print_image ( ditis ) :

    for n in range(len(ditis)) :
        _index = 0
        #print([ i for i in range(len(lables[n])) if lables[n][i] > 0. ])

        for i in range(28):
            _v2 = []
            for j in range(28) :

                if ditis[n][_index][0] > 0. :
                    value = ditis[n][_index][0]
                else :
                    value = ' '
                _v2.append(value)
                _index = _index + 1
            print(_v2)

def main () :
    loadAndPrint()


if __name__ == '__main__' :
    mnist = input_data.read_data_sets("../mnist/data/", one_hot=True)
    batch_xs, batch_ys = mnist.train.next_batch(10)
    #printMNIST ( batch_xs, batch_ys )

    IMG_PATH = '/home/mhkim/data/images/number_font.png'

    img = plt.imread(IMG_PATH, 'png')

    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()

    _image = tf.image.crop_to_bounding_box(img, 40, 35, 110, 70)

    _image = tf.image.resize_images(_image, [28, 28])

    rst = _image.eval()

    print_image ( [ [ x for l in rst for x in l ] ] )

    #printMNIST(batch_xs, batch_ys)

    #plt.imshow(batch_xs)

    #plt.show()
