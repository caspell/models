import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from utils import mnist_common as cmm
import matplotlib.gridspec as gridspec

class MnistCnn () :
    train_checkpoint = '/home/mhkim/data/checkpoint/mnist_cnn/save.ckpt'

    IMAGE_SIZE = 28
    NUM_CHANNELS = 1
    NUM_LABELS = 10
    SEED = 66478

    def __init__(self) :

        self.eval_data = tf.placeholder( tf.float32, shape=(1, self.IMAGE_SIZE, self.IMAGE_SIZE, self.NUM_CHANNELS), name='eval_data')

        W1 = tf.Variable(tf.truncated_normal([5, 5, self.NUM_CHANNELS, 32], stddev=0.1, seed=self.SEED, dtype=tf.float32), name='W1')
        b1 = tf.Variable(tf.zeros([32], dtype=tf.float32), name='bias1')

        W2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1, seed=self.SEED, dtype=tf.float32), name='W2')
        b2 = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32), name='bias2')

        fc1_weight = tf.Variable( tf.truncated_normal([self.IMAGE_SIZE // 4 * self.IMAGE_SIZE // 4 * 64, 512], stddev=0.1, seed=self.SEED, dtype=tf.float32), name='fc1_weight')
        fc1_bias = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32), name='fc1_bias')

        fc2_weight = tf.Variable(tf.truncated_normal([512, self.NUM_LABELS], stddev=0.1, seed=self.SEED, dtype=tf.float32), name='fc2_weight')
        fc2_bias = tf.Variable(tf.constant(0.1, shape=[self.NUM_LABELS], dtype=tf.float32), name='fc2_bias')

        self.W1 = W1
        self.B1 = b1
        self.W2 = W2
        self.B2 = b2
        self.FW1 = fc1_weight
        self.FW2 = fc2_weight
        self.FB1 = fc1_bias
        self.FB2 = fc2_bias

        with tf.name_scope('model'):
            conv = tf.nn.conv2d(self.eval_data, W1, strides=[1, 1, 1, 1], padding='SAME')

            self.conv1 = conv

            relu = tf.nn.relu(tf.nn.bias_add(conv, b1), name='relu1')

            self.relu1 = relu

            pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            self.pool1 = pool

            conv = tf.nn.conv2d(pool, W2, strides=[1, 1, 1, 1], padding='SAME')

            self.conv2 = conv

            relu = tf.nn.relu(tf.nn.bias_add(conv, b2))

            self.relu2 = relu

            pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            self.pool2 = pool

            pool_shape = pool.get_shape().as_list()

            reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

            hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(reshape, fc1_weight) + fc1_bias), 1., seed=self.SEED)

            logits = tf.matmul(hidden, fc2_weight) + fc2_bias

        #self.eval_prediction = tf.nn.softmax(logits)
        self.eval_prediction = logits

        # sess = tf.InteractiveSession()
        # init = tf.global_variables_initializer()
        # sess.run(init)
        # saver = tf.train.Saver()
        # saver.restore(sess, self.train_checkpoint)
        # sess.close()

    def scan (self, data) :

        print ( np.shape(data) )

        data_height , data_width , data_channel = np.shape(data)

        input_data = tf.placeholder(tf.float32, shape=(data_height , data_width , data_channel))

        W1 = tf.Variable(tf.truncated_normal([data_height, 5, data_channel, data_width / 5], stddev=0.1, seed=self.SEED, dtype=tf.float32))

        b1 = tf.Variable(tf.zeros([data_width / 5], dtype=tf.float32))

        sess = tf.InteractiveSession()

        conv1 = tf.nn.conv2d(input_data, W1, strides=[1,1,1,1], padding='SAME')

        result = conv1.eval({input_data : data})

        print (result)

        sess.close()

        return "not yet"


    def gridView(self, images) :

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

    def execute (self, data) :

        sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, self.train_checkpoint)

#        image = tf.placeholder("float", [None, None, 4])

        #_count , _, _ ,_ = np.shape(data)

        diff = 4 - np.ndim(data)

        for i in range(diff) :
            data = [data]

        returnValues = []

        for imgRst in data :

            _h , _w , _ = np.shape(imgRst)

            result = tf.argmax(tf.nn.softmax(self.eval_prediction), 1)

            resultValue = result.eval({self.eval_data: [imgRst]})[0]

            returnValues.append(resultValue)

        conv1W = self.conv1.eval({self.eval_data: [imgRst]})[0]

        conv2W = self.conv2.eval({self.eval_data: [imgRst]})[0]

        relu1W = self.relu1.eval({self.eval_data: [imgRst]})[0]

        relu2W = self.relu2.eval({self.eval_data: [imgRst]})[0]

        pool1W = self.pool1.eval({self.eval_data: [imgRst]})[0]

        pool2W = self.pool2.eval({self.eval_data: [imgRst]})[0]

        self.gridView([conv1W,conv2W])

        #self.gridView(conv2W)

        #self.gridView(relu1W)

        #self.gridView(relu2W)

        #self.gridView(pool1W)

        #self.gridView(pool2W)

        sess.close()


 #       slice = tf.slice(image, [0, 0, 0], [-1, -1, 1])

#        imgRst = sess.run(slice, feed_dict={image: data})

        ##imgOp = tf.image.resize_images(imgT, (185, 185))

        #imgRst = np.squeeze(imgRst, 2)

        #print(np.shape(data))
        #print(np.shape(imgRst))

  #      ictobb = tf.image.crop_to_bounding_box(imgRst, 0, 300, 185, 70)

        #print (np.shape(ictobb.eval()))

   #     riwcop = tf.image.resize_images(ictobb, (28, 28))

    #    imgRst = riwcop.eval()


        #plt.imshow(np.squeeze(imgRst, 2))
        #plt.show()

        #print ( np.shape(imgRst))


#         result = tf.argmax(tf.nn.softmax(self.eval_prediction), 1)
#
# #        print ( tf.nn.softmax(self.eval_prediction).eval({self.eval_data: [imgRst]}) )
#
#         resultValue = result.eval({self.eval_data: [imgRst]})[0]
#
#         sess.close()

        return returnValues

if __name__ == '__main__' :

    test_data_filename = cmm.maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = cmm.maybe_download('t10k-labels-idx1-ubyte.gz')

    test_data = cmm.extract_data(test_data_filename, 100)
    test_labels = cmm.extract_labels(test_labels_filename, 1)


    # for _img in test_data :
    #     mnistCnn = MnistCnn()
    #     resultValue = mnistCnn.execute(_img)
    #     print ( resultValue )
    #     plt.imshow(np.squeeze(_img, 2))
    #     plt.show()



#    img = cmm.get_image()

    #print ( np.shape(img))

    #test_data = cmm.pack(test_data)

    cmm.packShow(test_data)

    #mnistCnn = MnistCnn()

    #resultValue = mnistCnn.scan(test_data)
    #resultValue = mnistCnn.execute(test_data)

    #print ( resultValue )

    #cmm.packShow(test_data)