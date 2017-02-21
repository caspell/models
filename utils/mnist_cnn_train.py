import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from utils import mnist_common as cmm
import time
import os
import sys
import math
import time

class MnistCnn :

    #train_checkpoint = './checkpoint/mnist_cnn/save.ckpt'
    train_checkpoint = '/home/mhkim/data/checkpoint/mnist_cnn/'

    IMAGE_SIZE = 28
    NUM_CHANNELS = 1
    NUM_LABELS = 10
    SEED = 66478
    PIXEL_DEPTH = 255
    VALIDATION_SIZE = 5000
    SEED = 66478
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    EVAL_FREQUENCY = 100  # Number of steps between evaluations.

    def __init__(self) :

        train_data_filename = cmm.maybe_download('train-images-idx3-ubyte.gz')
        train_labels_filename = cmm.maybe_download('train-labels-idx1-ubyte.gz')
        test_data_filename = cmm.maybe_download('t10k-images-idx3-ubyte.gz')
        test_labels_filename = cmm.maybe_download('t10k-labels-idx1-ubyte.gz')

        # Extract it into numpy arrays.
        self.train_data = cmm.extract_data(train_data_filename, 60000)
        self.train_labels = cmm.extract_labels(train_labels_filename, 60000)

        self.test_data = cmm.extract_data(test_data_filename, 10000)
        self.test_labels = cmm.extract_labels(test_labels_filename, 10000)

        zeros1 = np.zeros_like(self.train_data[0])
        ones1 = np.ones_like(self.train_data[0])
        fills = np.full_like(self.train_data[0], -0.5)

        self.train_data[0] = zeros1
        self.train_data[1] = ones1
        self.train_data[2] = fills


        self.train_labels[0] = -1
        self.train_labels[1] = -1
        self.train_labels[2] = -1

        # Generate a validation set.
        # validation_data = train_data[:self.VALIDATION_SIZE, ...]
        # validation_labels = train_labels[:self.VALIDATION_SIZE]


        self.W1 = tf.Variable(tf.truncated_normal([5, 5, self.NUM_CHANNELS, 32], stddev=0.1, seed=self.SEED, dtype=tf.float32), name='W1')
        self.W2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1, seed=self.SEED, dtype=tf.float32), name='W2')

        self.b1 = tf.Variable(tf.zeros([32], dtype=tf.float32), name='bias1')
        self.b2 = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32), name='bias2')

        self.fc1_weight = tf.Variable( tf.truncated_normal([self.IMAGE_SIZE // 4 * self.IMAGE_SIZE // 4 * 64, 512], stddev=0.1, seed=self.SEED, dtype=tf.float32), name='fc1_weight')
        self.fc1_bias = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32), name='fc1_bias')

        self.fc2_weight = tf.Variable(tf.truncated_normal([512, self.NUM_LABELS], stddev=0.1, seed=self.SEED, dtype=tf.float32), name='fc2_weight')
        self.fc2_bias = tf.Variable(tf.constant(0.1, shape=[self.NUM_LABELS], dtype=tf.float32), name='fc2_bias')

    def model (self, data, name='model', train=False):
        with tf.name_scope(name):
            conv = tf.nn.conv2d(data, self.W1, strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, self.b1), name='relu1')
            pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            conv = tf.nn.conv2d(pool, self.W2, strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, self.b2))
            pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            pool_shape = pool.get_shape().as_list()

            reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

            hidden = tf.nn.relu(tf.matmul(reshape, self.fc1_weight) + self.fc1_bias)

            if train:
                hidden = tf.nn.dropout(hidden, 0.5, seed=self.SEED)

            return tf.matmul(hidden, self.fc2_weight) + self.fc2_bias

    def train (self) :

        if tf.gfile.Exists(self.train_checkpoint):
            tf.gfile.DeleteRecursively(self.train_checkpoint)
        tf.gfile.MakeDirs(self.train_checkpoint)


        train_data = self.train_data[self.VALIDATION_SIZE:, ...]
        train_labels = self.train_labels[self.VALIDATION_SIZE:]

        train_size = train_labels.shape[0]

        train_data_node = tf.placeholder(tf.float32, shape=(self.BATCH_SIZE, self.IMAGE_SIZE, self.IMAGE_SIZE, self.NUM_CHANNELS),
                                         name="train_data_node")
        train_labels_node = tf.placeholder(tf.int64, shape=(self.BATCH_SIZE,), name="train_labels_node")

        logits = self.model(train_data_node , 'model', True)

        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=train_labels_node),
                              name='cost')

        regularizers = (
        tf.nn.l2_loss(self.fc1_weight) + tf.nn.l2_loss(self.fc1_bias) + tf.nn.l2_loss(self.fc2_weight) + tf.nn.l2_loss(
            self.fc2_bias))

        cost += 5e-4 * regularizers

        batch = tf.Variable(0, dtype=tf.float32, name='batch')

        learning_rate = tf.train.exponential_decay(0.01, batch * self.BATCH_SIZE, train_size, 0.95, staircase=True,
                                                   name='decay_learning_rate')

        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cost, global_step=batch, name='optimizer')

        train_prediction = tf.nn.softmax(logits, name='train_prediction')

        start_time = time.time()

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            saver = tf.train.Saver()

            for step in range(int(self.NUM_EPOCHS * train_size) // self.BATCH_SIZE):
                offset = (step * self.BATCH_SIZE) % (train_size - self.BATCH_SIZE)
                batch_data = train_data[offset: (offset + self.BATCH_SIZE)]
                batch_labels = train_labels[offset:(offset + self.BATCH_SIZE)]

                feed_dict = {train_data_node: batch_data, train_labels_node: batch_labels}

                sess.run(optimizer, feed_dict=feed_dict)

                if step % self.EVAL_FREQUENCY == 0:
                #    l, lr, predictions = sess.run([cost, learning_rate, train_prediction], feed_dict=feed_dict)
                    elapsed_time = time.time() - start_time
                    #start_time = time.time()
                    print ('Step %d , %.2f s' % (step, elapsed_time))

            elapsed_time = time.time() - start_time

            print('total time : %.2f s' % elapsed_time)

            saver.save(sess=sess, save_path=os.path.join(self.train_checkpoint, 'save.ckpt'))

            sys.stdout.flush()

            print("finished!")

    def execute (self, data) :

        diff = 4 - np.ndim(data)

        for i in range(diff):
            data = [data]

        batch_size = np.shape(data)[0]

        eval_data = tf.placeholder(tf.float32, shape=(batch_size, self.IMAGE_SIZE, self.IMAGE_SIZE, self.NUM_CHANNELS),
                                   name='eval_data')

        # returnValues = []
        #
        # for imgRst in data:
        #     _h, _w, _ = np.shape(imgRst)
        #
        #     result = tf.argmax(tf.nn.softmax(self.eval_prediction), 1)
        #
        #     resultValue = result.eval({self.eval_data: [imgRst]})[0]
        #
        #     returnValues.append(resultValue)

        logits = self.model(eval_data, 'model')

        eval_prediction = tf.nn.softmax(logits)

        sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(self.train_checkpoint, 'save.ckpt'))

        result = tf.argmax(eval_prediction, 1)

        logitResult = logits.eval({eval_data: data})
        predictResult = eval_prediction.eval({eval_data: data})

        resultValue = result.eval({eval_data: data})

        # for i in range(batch_size) :
        #     print('--------------------------------------------------------------------------------------')
        #     print (resultValue[i])
        #     print(['%.2f' % x for x in logitResult[i]])
        #     print(['%.8f' % x for x in predictResult[i]])
        #     print('--------------------------------------------------------------------------------------')

        #self.showImage(data)

        sess.close()

        return resultValue

    def getSoftmaxValuesHistogram (self, data) :

        diff = 4 - np.ndim(data)

        for i in range(diff):
            data = [data]

        batch_size = np.shape(data)[0]

        eval_data = tf.placeholder(tf.float32, shape=(batch_size, self.IMAGE_SIZE, self.IMAGE_SIZE, self.NUM_CHANNELS))

        logits = self.model(eval_data, 'model')

        # eval_prediction = tf.nn.softmax(logits)

        init = tf.global_variables_initializer()

        sess = tf.InteractiveSession()
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(self.train_checkpoint, 'save.ckpt'))

        logitResult = logits.eval({eval_data: data})
        # predictResult = eval_prediction.eval({eval_data: data})

        # for i in range(batch_size) :
        #     if np.max(logitResult[i]) < 10 :
        #         print('--------------------------------------------------------------------------------------')
        #         print ('index %d' %i)
        #         print (np.max(logitResult[i]))
        #         print(['%.2f' % x for x in logitResult[i]])
        #         print('--------------------------------------------------------------------------------------')

        resultValue = [np.max(logitResult[i]) for i in range(batch_size)]

        sess.close()

        return resultValue

    def test (self, data):

        h , w , c = np.shape(data)

        input = tf.placeholder(dtype=tf.float32, shape=(1, h , w , c), name='input1')

        pool = tf.nn.max_pool(input, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='SAME')
        #pool = tf.nn.max_pool(pool, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # pool = tf.nn.max_pool(pool, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        init = tf.global_variables_initializer()
        sess = tf.InteractiveSession()
        sess.run(init)

        result = pool.eval({input:[data]})

        # print ( result )

        sess.close();

        print (np.shape(result[0]))

        plt.imshow(np.squeeze(result[0], 2))
        plt.show()

    def find3 (self, data) :

        CHAR_WIDTH = 28

        print ( np.shape(data) )

        h , w , c = np.shape(data)

        rate = 2

        base_size = CHAR_WIDTH * rate
        _y = math.ceil(h / CHAR_WIDTH / rate)
        _x = math.ceil(w / CHAR_WIDTH / rate)

        h , w =  _y * base_size , _x * base_size

        canvas = cmm.getCanvas(h , w)

        data = cmm.imageCopy(canvas, data)

        size = _y * _x

        input = tf.placeholder(dtype=tf.float32, shape=(size , base_size , base_size , c), name='input1')

        images = []

        positions = []

        for i in range(size):

            offset = ((i // _x) * base_size, (i % _x) * base_size)

            positions.append(offset)

            item = data[offset[0]:offset[0]+base_size , offset[1] : offset[1]+base_size, :]

            images.append(item)

        images = np.asarray(images)

        def conv2 () :
            conv = tf.nn.conv2d(input , self.W1, strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, self.b1))
            pool = tf.nn.max_pool(relu, ksize=[1, 2 * rate , 2 * rate , 1], strides=[1, 2 * rate, 2 * rate, 1], padding='SAME')

            conv = tf.nn.conv2d(pool, self.W2, strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, self.b2))
            pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            return pool

        pool = conv2()

        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, self.fc1_weight) + self.fc1_bias)

        logits = tf.matmul(hidden, self.fc2_weight) + self.fc2_bias

        sft = tf.nn.softmax(logits)

        am = tf.argmax(sft , 1)
        # reduction_indices=[1]

        #eval = tf.reduce_max(sft)

        eval = sft

        init = tf.global_variables_initializer()
        sess = tf.InteractiveSession()
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(self.train_checkpoint, 'save.ckpt'))

        result = sess.run([eval], feed_dict={input: images})

        print ( np.shape(result[0]))

        sess.close()

        return ( np.squeeze(result, axis=2) , positions , base_size)

    def find2 (self, data) :

        CHAR_WIDTH = 28

        print ( np.shape(data) )

        h , w , c = np.shape(data)

        rate = 2

        base_size = CHAR_WIDTH * rate
        _y = math.ceil(h / CHAR_WIDTH / rate)
        _x = math.ceil(w / CHAR_WIDTH / rate)

        h , w =  _y * base_size , _x * base_size

        canvas = cmm.getCanvas(h , w)

        data = cmm.imageCopy(canvas, data)

        size = _y * _x

        input = tf.placeholder(dtype=tf.float32, shape=(size , base_size , base_size , c), name='input1')

        images = []

        positions = []

        for i in range(size):

            offset = ((i // _x) * base_size, (i % _x) * base_size)

            positions.append(offset)

            item = data[offset[0]:offset[0]+base_size , offset[1] : offset[1]+base_size, :]

            images.append(item)

        images = np.asarray(images)

        conv = tf.nn.conv2d(input , self.W1, strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, self.b1))
        pool = tf.nn.max_pool(relu, ksize=[1, 2 * rate , 2 * rate , 1], strides=[1, 2 * rate, 2 * rate, 1], padding='SAME')

        conv = tf.nn.conv2d(pool, self.W2, strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, self.b2))
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, self.fc1_weight) + self.fc1_bias)

        logits = tf.matmul(hidden, self.fc2_weight) + self.fc2_bias

        sft = tf.nn.softmax(logits)

        am = tf.argmax(sft , 1)
        # reduction_indices=[1]

        #eval = tf.reduce_max(sft)

        eval = sft

        init = tf.global_variables_initializer()
        sess = tf.InteractiveSession()
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(self.train_checkpoint, 'save.ckpt'))

        result = sess.run([eval], feed_dict={input: images})

        print ( np.shape(result[0]))

        sess.close()

        return ( np.squeeze(result, axis=2) , positions , base_size)

    def find1 (self, data) :

        CHAR_WIDTH = 28

        h , w , c = np.shape(data)

        base_size = CHAR_WIDTH * 2

        _y = math.ceil(h / CHAR_WIDTH / 2)

        _x = math.ceil(w / CHAR_WIDTH / 2)

        h , w =  _y * base_size , _x * base_size

        canvas = cmm.getCanvas(h , w)

        data = cmm.imageCopy(canvas, data)

        size = _y * _x

        input = tf.placeholder(dtype=tf.float32, shape=(h , w , c), name='input1')

        A = tf.TensorArray(tf.float32, size=size)
        #A = tf.TensorArray(tf.int64, size=size)

        positions = []

        for i in range(size) :

            offset = ((i // _x) * base_size, (i % _x) * base_size)

            positions.append(offset)

            tfImage = tf.image.crop_to_bounding_box(input, offset[0] , offset[1], base_size, base_size)

            tfImage = tf.reshape(tfImage, [1, base_size , base_size , c])

            # imagePooling = tf.nn.max_pool(tfImage, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # pool_shape = imagePooling.get_shape().as_list()
            # reshape = tf.reshape(imagePooling, [pool_shape[0], pool_shape[1] , pool_shape[2], 1])
            # conv = tf.nn.conv2d(reshape , self.W1, strides=[1, 1, 1, 1], padding='SAME')
            # relu = tf.nn.relu(tf.nn.bias_add(conv, self.b1))
            # pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            conv = tf.nn.conv2d(tfImage , self.W1, strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, self.b1))
            pool = tf.nn.max_pool(relu, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

            conv = tf.nn.conv2d(pool, self.W2, strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, self.b2))
            pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            pool_shape = pool.get_shape().as_list()
            reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
            hidden = tf.nn.relu(tf.matmul(reshape, self.fc1_weight) + self.fc1_bias)

            logits = tf.matmul(hidden, self.fc2_weight) + self.fc2_bias

            sft = tf.nn.softmax(logits)

            am = tf.argmax(sft , 1)
            # reduction_indices=[1]
            eval = tf.reduce_max(sft)
            # tf.cond(x )

            A = A.write(i, sft)

        d = A.pack()

        init = tf.global_variables_initializer()
        sess = tf.InteractiveSession()
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(self.train_checkpoint, 'save.ckpt'))


        result = sess.run([d], feed_dict={input:data})

        # print ( result )

        sess.close()

        return ( np.squeeze(result, axis=2) , positions , base_size)

    def parseImage (self, image) :

        y , x , _ = np.shape(image)

        if y > x : basic = x
        else : basic = y

        cell = int(basic / 10)

        if x % cell > 0 :
            bg = cmm.getCanvas(y, (x / cell + 1) * cell)
            image = cmm.imageCopy(bg , image)

        if y % cell > 0 :
            bg = cmm.getCanvas(int(y / cell + 1) * cell, x)
            image = cmm.imageCopy(bg, image)

        y, x, _ = np.shape(image)

        return ( image , cell , int ( y / cell * x / cell ) )

    def pulv (self, image , cell , batch_size):
        pass

def main1() :

    test_data_filename = cmm.maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = cmm.maybe_download('t10k-labels-idx1-ubyte.gz')

    test_data = cmm.extract_data(test_data_filename, 100)
    test_labels = cmm.extract_labels(test_labels_filename, 100)

    mnistCnn = MnistCnn()

    test_data1 = cmm.pack2(test_data[0:10])

    test_data2 = cmm.pack2(test_data[10:20])

    test_data3 = cmm.pack2(test_data[20:35], 2.)

    #origin = cmm.getCanvas(480 , 640)

    image = "/home/mhkim/사진/dream_d9e49f73ad.jpg"

    origin = cmm.getCanvas2(512, 850)
    #origin = cmm.getCanvas2(512, 850 , image=image)
    # print ( np.shape(origin) )

    origin = cmm.imageCopy2(origin, test_data1)

    origin = cmm.imageCopy2(origin, test_data2, (20, 40))

    origin = cmm.imageCopy2(origin, test_data3, (420, 0))

    plt.imshow(origin)
    plt.show()

    origin = cmm.parse_image(origin)

    data = (values , offsets , rect_size) = mnistCnn.find1(origin)
    cmm.showImageGrid(origin , data , format='%.2f', rate=0.55, color='white')

def main2() :

    test_data_filename = cmm.maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = cmm.maybe_download('t10k-labels-idx1-ubyte.gz')

    test_data = cmm.extract_data(test_data_filename, 100)
    test_labels = cmm.extract_labels(test_labels_filename, 100)

    # cmm.showImage2(test_data)

    mnistCnn = MnistCnn()
    #

    test_data1 = cmm.pack2(test_data[0:10], spacing=1.)

    test_data2 = cmm.pack2(test_data[20:30], spacing=1.)

    test_data3 = cmm.pack2(test_data[30:40], spacing=1. , power=2.)

    image = "/home/mhkim/사진/dream_d9e49f73ad.jpg"

    #origin = cmm.getCanvas2(28 * 10, 28 * 20, image=image)

    origin = cmm.getCanvas2(28 * 10, 28 * 20, image=image)

    origin = cmm.imageCopy2(origin, test_data1, (0, 0))

    origin = cmm.imageCopy2(origin, test_data2, (28 * 3, 0))

    origin = cmm.imageCopy2(origin, test_data3, (28 * 6, 0))

    origin = cmm.parse_image(origin)

    print(np.shape(origin))

    origin = origin / [-255 * 2]
    #
    # plt.imshow(np.squeeze(origin))
    # plt.show()

    data = (values, offsets, rect_size) = mnistCnn.find2(origin)

    cmm.showImageGrid(origin, data, format='%.2f', rate='each', color='white')

def main3():

    test_data_filename = cmm.maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = cmm.maybe_download('t10k-labels-idx1-ubyte.gz')

    test_data = cmm.extract_data(test_data_filename, 100)
    test_labels = cmm.extract_labels(test_labels_filename, 100)

    # cmm.showImage2(test_data)

    mnistCnn = MnistCnn()
    #

    test_data1 = cmm.pack2(test_data[0:10], spacing=1.)

    test_data2 = cmm.pack2(test_data[20:30], spacing=1.)

    test_data3 = cmm.pack2(test_data[30:40], spacing=1. , power=2.)

    image = "/home/mhkim/사진/dream_d9e49f73ad.jpg"

    #origin = cmm.getCanvas2(28 * 10, 28 * 20, image=image)

    origin = cmm.getCanvas2(28 * 10, 28 * 20, image=image)

    origin = cmm.imageCopy2(origin, test_data1, (0, 0))

    origin = cmm.imageCopy2(origin, test_data2, (28 * 3, 0))

    origin = cmm.imageCopy2(origin, test_data3, (28 * 6, 0))

    origin = cmm.parse_image(origin)

    print(np.shape(origin))

    origin = origin / [-255 * 2]
    #
    # plt.imshow(np.squeeze(origin))
    # plt.show()

    data = (values, offsets, rect_size) = mnistCnn.find3(origin)

    cmm.showImageGrid(origin, data, format='%.2f', rate='each', color='white')

if __name__ == '__main__' :
    main3()