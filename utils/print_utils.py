import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../mnist/data/", one_hot=True)

batch_xs , batch_ys = mnist.train.next_batch(10)
test_xs , test_ys = mnist.test.next_batch(1)

#print(len(batch_xs[0]))

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


#printMNIST ( batch_xs , batch_ys )

print ( '{:_<150}\n' .format('_') )

printMNIST ( test_xs , test_ys )