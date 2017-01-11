import tensorflow as tf
from os import listdir
from os.path import isfile, join

data_path = './data-csv'
filenames = [join(data_path,f) for f in listdir(data_path) if isfile(join(data_path,f))]

import numpy as np

block_size = 10
seq_len = 8
batch_size = 5
queue_capacity = 32
num_threads = 4
ncols = 3

filename_queue = tf.train.string_input_producer(filenames)
block = tf.TextLineReader().read_up_to(filename_queue, block_size).values
subsequences = [tf.slice(block, [i], [block_size - seq_len + 1]) for i in range(seq_len)]
batched = tf.train.shuffle_batch(subsequences,
                                 batch_size, 
                                 capacity=queue_capacity,
                                 min_after_dequeue=queue_capacity/2,     
                                 num_threads=num_threads,
                                 enqueue_many=True)
decoded = [tf.pack(tf.decode_csv(b, [[0.]] * ncols, ','),axis=1) for b in batched]
decoded = tf.pack(decoded,axis=1)

print(decoded.get_shape())

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord)

    for x in range(3):
        y = sess.run([decoded])
        print('---')
        for z in y[0]:
            print(z)

    coord.request_stop()