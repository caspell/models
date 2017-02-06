import numpy as np
import tensorflow as tf

z1 = np.zeros((1, 10), dtype=np.float32)

print (z1)

sess = tf.InteractiveSession()

q = tf.FIFOQueue(10, "float")

init = q.enqueue_many(z1)

x = q.dequeue()

y = x + 1

q_inc = q.enqueue([y])

init.run(session=sess)

q_inc.run()
q_inc.run()
q_inc.run()
q_inc.run()

sess.close()