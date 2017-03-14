# from tensorflow.python.client import device_lib
#
# print ( device_lib.list_local_devices() )

from matplotlib import pyplot as plt
import time
import os, sys, cv2

f = plt.figure()
ax = f.gca()
f.show()

for i in range(100):
    im = cv2.imread('/data/share/nfs/40/latest.jpg')
    ax.imshow(im)
    f.canvas.draw()
    #raw_input('pause : press any key ...')
    time.sleep(0.3)
