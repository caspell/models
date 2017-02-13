import tensorflow as tf

cluster = tf.train.ClusterSpec({"local":['192.168.1.50:2222']})
server = tf.train.Server(cluster, job_name="local", task_index=0)
server.join()