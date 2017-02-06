import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#import print_utils as pu

IMG_PATH = '/home/mhkim/data/images/number_font.png'

img = plt.imread(IMG_PATH, 'png')

#print ( img )





sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

_image = tf.image.crop_to_bounding_box(img , 40, 35, 110 ,70)

_image = tf.image.resize_images(_image, [28, 28])

rst = _image.eval()

print ( len(rst[0][0]) )

#plt.imshow(rst)

#plt.show()

#pu.print_image (rst)

sess.close()
