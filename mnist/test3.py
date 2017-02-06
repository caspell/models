import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

xy = np.loadtxt('test3.txt', unpack=True, dtype='float32')

x_data = xy[0:-1]

y_data = xy[-1]

#print (x_data)
#print (y_data)

X = tf.placeholder(dtype=tf.float32)
Y = tf.placeholder(dtype=tf.float32)

#print (len(x_data))

W = tf.Variable(tf.random_uniform([1,len(x_data)], -1.0 , 1.0 ))

h = tf.matmul(W, X)

#hypothesis = tf.div(1. , 1. + tf.exp(-h))
#
#hypothesis = tf.sigmoid(h)
hypothesis = tf.nn.softmax(h)

#cost = tf.reduce_mean(tf.square(hypothesis - y_data))
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + ( 1 - Y ) * tf.log(1 - hypothesis) )

learning_rate = tf.Variable(0.1)
train = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate
).minimize(cost)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

result1 = []
result2 = []
result3 = []
for step in range(200) :

    sess.run(train, feed_dict={ X:x_data, Y:y_data })
    #sess.run(train)

#    result1.append(sess.run(cost))
  #  result2.append(sess.run(W)[0])

print ( sess.run(hypothesis, feed_dict={ X:[[1 , 1],[5 , 3],[5 , 4]] }) )

#plt.subplot(2,1,1)
#plt.plot(result1)

#plt.subplot(2,1,2)
#plt.plot(result2)

#plt.show()

print( tf.nn.softmax([1.,4.,5.]).eval()  )

sess.close()