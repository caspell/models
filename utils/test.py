import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from utils import mnist_common as cmm
import matplotlib.gridspec as gridspec


test_data_filename = cmm.maybe_download('t10k-images-idx3-ubyte.gz')
test_labels_filename = cmm.maybe_download('t10k-labels-idx1-ubyte.gz')

test_data = cmm.extract_data(test_data_filename, 10)
test_labels = cmm.extract_labels(test_labels_filename, 1)


# for _img in test_data :
#     mnistCnn = MnistCnn()
#     resultValue = mnistCnn.execute(_img)
#     print ( resultValue )
#     plt.imshow(np.squeeze(_img, 2))
#     plt.show()



#    img = cmm.get_image()

#print ( np.shape(img))

test_data = cmm.pack(test_data)

#cmm.packShow(test_data)


print ( np.shape(test_data) )
