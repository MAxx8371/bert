import tensorflow as tf

# embedding = tf.constant([[1,2],[2,3],[3,4],[4,5],[5,6]])
# ids = tf.constant([1,2,4,0,1])
# assert_op = tf.assert_less_equal(2.0,1.0)

# with tf.Session() as sess:
#     print(sess.run(assert_op))

x = 1.4
y = 1.3
sess = tf.Session()
with tf.control_dependencies([tf.assert_less_equal(x, y)]):
  output = tf.reduce_sum(x)
  print(sess.run(output))