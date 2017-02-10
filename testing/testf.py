import tensorflow as tf
import numpy as np
from utils import mnist_common as cmm
import matplotlib.pyplot as plt
import matplotlib.image as mimage
import os
from PIL import Image, ImageFilter

train_data_filename = cmm.maybe_download('train-images-idx3-ubyte.gz')
train_labels_filename = cmm.maybe_download('train-labels-idx1-ubyte.gz')
test_data_filename = cmm.maybe_download('t10k-images-idx3-ubyte.gz')
test_labels_filename = cmm.maybe_download('t10k-labels-idx1-ubyte.gz')


train_data = cmm.extract_data(train_data_filename, 1)
train_labels = cmm.extract_labels(train_labels_filename, 1)


#train_data = np.squeeze(train_data[0], axis=2)

#print ( np.squeeze(train_data, axis=np.ndim(train_data) - 1))

#print ( np.shape(np.squeeze(train_data, axis=np.ndim(train_data) - 1)) )

#train_data = np.clip(train_data, 0, 255)

train_data = train_data[0]

#train_data[train_data<0]=0

#print ( train_data)

#plt.imshow(np.squeeze(train_data, axis=2))
#plt.show()

nz = np.zeros((3,4))

print ( np.shape(nz) )
print ( nz )



imageDir = '/home/mhkim/data/images'

#imagePath = os.path.join(imageDir, 'number_font.png')

imagePath = "/home/mhkim/사진/dream_0de4cd8b81.jpg"

img = Image.open(imagePath)
''''''
#img = mimage.imread(imagePath)


def masking ( img ) :
    mask = img.convert("L")
    return mask.point(lambda i : i < 100 and 255)

plt.imshow(masking(img))
plt.show()


img1 = img.filter(ImageFilter.EMBOSS)

img2 = img.filter(ImageFilter.FIND_EDGES)

img1 = masking(img1)
img2 = masking(img2)

print( np.shape(img1) )
print( np.shape(img2) )

fig = plt.figure()
fig.add_subplot(1,2,1).set_title('test1')
plt.imshow(img1)

fig.add_subplot(1,2,2).set_title('test2')
plt.imshow(img2)

plt.show()

