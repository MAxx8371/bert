import tensorflow as tf
import re

# embedding = tf.constant([[1,2],[2,3],[3,4],[4,5],[5,6]])
# ids = tf.constant([1,2,4,0,1])
# assert_op = tf.assert_less_equal(2.0,1.0)

# with tf.Session() as sess:
#     print(sess.run(assert_op))

input_files = []
input_file = "E:\\code\\bert\\demo.py,demo.py,pwd.cpp"
for input_pattern in input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

flat_offsets = tf.reshape(
      tf.range(0, 2, dtype=tf.int32) * 8, [-1, 1]) 

positions = tf.constant([[1,2,3],[4,5,6]])
flat_positions = tf.reshape(positions + flat_offsets, [-1])
# :  索引位置 [batch_size, masked_len]

a = tf.constant(1,shape=[16,2],dtype=tf.float32, name='v1')
output_tensor = tf.gather(a, flat_positions)


filenames = ["./file1.txt", "./file2.txt"]
dataset = (tf.data.Dataset.from_tensor_slices(filenames)
           .parallel_interleave(lambda x:
               tf.data.TextLineDataset(x),
               sloppy=True,
               cycle_length=2))

iterator = dataset.make_one_shot_iterator().get_next()
num_batch = 0

with tf.Session() as sess:
    print(sess.run(output_tensor))

    while True:
        try:
            value = sess.run(iterator)
            print(value)
            num_batch += 1
        except tf.errors.OutOfRangeError:
            break