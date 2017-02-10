import tensorflow as tf
import time
from cnn_train_clustering import MNISTtrain

# tf.train.ClusterSpec 정의를 위한 설정
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")

# tf.train.Server 정의를 위한 설정
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS


def main(_):

    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # 변수 서버와 작업자 클러스터를 생성한다.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # 로컬 작업 수행을 위해 서버를 생성하고 구동한다.
    server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

    mnistTrain = MNISTtrain()

    loop_count = mnistTrain.get_batch_size()

    if FLAGS.job_name == "ps":
        server.join()

    elif FLAGS.job_name == "worker":

        # 작업자에 연산을 분배한다.
        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):

            # 모델 구축...
            #loss = mnistTrain.get_loss()

            global_step = tf.Variable(0, dtype=tf.float32, name='batch')

            train_op = mnistTrain.get_train()

            saver = tf.train.Saver()

            summary_op = tf.merge_all_summaries()

            #init_op = tf.initialize_all_variables()
            init_op = tf.global_variables_initializer()

            # 훈련 과정을 살펴보기 위해 "supervisor"를 생성한다.
            sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                     logdir="/tmp/train_logs",
                                     init_op=init_op,
                                     summary_op=summary_op,
                                     saver=saver,
                                     global_step=global_step,
                                     save_model_secs=600)

        # supervisor는 세션 초기화를 관리하고, checkpoint로부터 모델을 복원하고
        # 에러가 발생하거나 연산이 완료되면 프로그램을 종료한다.

        start_time = time.time()

        with sv.managed_session(server.target) as sess:

            for step in range(loop_count):

                if sv.should_stop() : break

                offset = (step * MNISTtrain.BATCH_SIZE) % (MNISTtrain.train_size - MNISTtrain.BATCH_SIZE)
                batch_data = MNISTtrain.train_data[offset: (offset + MNISTtrain.BATCH_SIZE)]
                batch_labels = MNISTtrain.train_labels[offset:(offset + MNISTtrain.BATCH_SIZE)]

                feed_dict = {MNISTtrain.train_data_node: batch_data, MNISTtrain.train_labels_node: batch_labels}

                _, step = sess.run([train_op, global_step] , feed_dict=feed_dict)

                if step % MNISTtrain.EVAL_FREQUENCY == 0:
                    l, lr, predictions = sess.run([MNISTtrain.cost, MNISTtrain.learning_rate, MNISTtrain.train_prediction], feed_dict=feed_dict)
                    elapsed_time = time.time() - start_time
                    start_time = time.time()

                    print('--------------------------------------------------------------------')
                    print('Step %d (epoch %.2f), %.1f ms' % (
                      step, float(step) * MNISTtrain.BATCH_SIZE / MNISTtrain.train_size, 1000 * elapsed_time / MNISTtrain.EVAL_FREQUENCY))
                    print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))

                    print('Step %d (epoch %.2f), %.1f ms' % (
                    step, float(step) * MNISTtrain.BATCH_SIZE / MNISTtrain.train_size, 1000 * elapsed_time / MNISTtrain.EVAL_FREQUENCY))

        sv.stop()

if __name__ == "__main__":
    tf.app.run()