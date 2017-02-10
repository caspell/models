import tensorflow as tf

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

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # 작업자에 연산을 분배한다.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      # 모델 구축...
      loss = ...
      global_step = tf.Variable(0)

      train_op = tf.train.AdagradOptimizer(0.01).minimize(
          loss, global_step=global_step)

      saver = tf.train.Saver()
      summary_op = tf.merge_all_summaries()
      init_op = tf.initialize_all_variables()

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
    with sv.managed_session(server.target) as sess:
      # "supervisor"가 종료되거나 1000000 step이 수행 될 때까지 반복한다.
      step = 0
      while not sv.should_stop() and step < 1000000:
        # 훈련 과정을 비동기식으로 실행한다.Run a training step asynchronously.
        # 동기식 훈련 수행을 위해서는 `tf.train.SyncReplicasOptimizer`를 참조하라.
        _, step = sess.run([train_op, global_step])

    # 모든 서비스 중단.
    sv.stop()

if __name__ == "__main__":
  tf.app.run()