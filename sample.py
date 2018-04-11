import config
import tensorflow as tf
import sys
import numpy as np
from tensorflow.contrib import rnn
import functions


def sample(file_name, start):
    vocabulary_size = config.get("vocabulary_size")
    num_hidden = config.get("num_hidden")
    batch_size = config.get("batch_size")
    layers = config.get("layers")
    num_outputs = config.get("num_outputs")


    vocabulary=functions.load_vocabulary()

    with tf.Graph().as_default() as g:

        sample_input = tf.placeholder(tf.int32, shape=(batch_size, 1))
        inputs = tf.unstack(sample_input, 1, 1)

        inputs_encoded = [tf.one_hot(elem, depth=vocabulary_size) for elem in inputs]

        rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(num_units=num_hidden, forget_bias=1.0) for _ in range(layers)], state_is_tuple=True)
        state = rnn_cell.zero_state(batch_size, tf.float32)

        outputs, after_state = rnn.static_rnn(rnn_cell, inputs_encoded, initial_state=state, dtype=tf.float32)

        flat_outputs = tf.reshape(tf.concat(axis=1, values=outputs), [-1, num_hidden])

        softmax_w = tf.Variable(tf.random_normal([num_hidden, num_outputs]))
        softmax_b = tf.Variable(tf.random_normal([vocabulary_size]))

        logits = tf.matmul(flat_outputs, softmax_w) + softmax_b

        init_op = tf.initialize_all_variables()
        saver = tf.train.Saver()

        with tf.Session(graph=g) as sess:
            sess.run(init_op)
            saver.restore(sess, file_name)

            res = []
            id = 0
            zero_state = rnn_cell.zero_state(batch_size, tf.float32)
            cur_state = sess.run(zero_state)
            for i in range(0, config.get('sample_length')):
                feeded_input = np.zeros((1, 1), dtype=np.int32)
                if i<len(start):
                    feeded_input[0][0] = start[i]
                else:
                    feeded_input[0][0] = id

                res.append(vocabulary[feeded_input[0][0]])

                lgts,cur_state = sess.run([tf.nn.softmax(logits),after_state], feed_dict={sample_input:feeded_input, state:cur_state})
                lgts = lgts.reshape((batch_size, 1, vocabulary_size))
                #id = np.argmax(lgts[0][0])
                id = np.random.choice(vocabulary_size, 1, p=lgts[0][0])[0]

            print("".join(res))


if __name__ == '__main__':
    if len(sys.argv) == 2:
        dump_file = sys.argv[1]
    else:
        print("Usage: sample.py model_save.ckpt")
        exit(-1)

    config.load("train.json")
    config.set("batch_size",1)

    sample(dump_file, [37])