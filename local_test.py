import tensorflow as tf
c = tf.constant("Hello, distributed tensorflow!")
server =tf.train.Server.create_local_server()

sess = tf.Session(server.target)

print(server.target)

print ( sess.run(c))

sess.close()

#tf.train.ClusterSpec 생성

# tf.train.Server
'''
clusterSpec = tf.train.ClusterSpec({
    "worker":[ "worker0:2222" , "worker1:2222", "worker2:12222"]
    , "ps":[ "ps0:2222", "ps2:2222" ]
})


tf.train.Server(clusterSpec)

'''