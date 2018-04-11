import config
import sys
import tensorflow as tf
from tensorflow.contrib import rnn
import functions



def next_data_batch():
    seq_length = config.get("time_steps")
    batch_size = config.get("batch_size")
    def gen():
        p = 0
        while p + seq_length + 1 < len(data):
            yield (encoded_data[p:p + seq_length], encoded_data[p + 1:p + 1 + seq_length])
            p += 1

    ds = tf.data.Dataset.from_generator(gen, (tf.int32, tf.int32), ((seq_length,), (seq_length,)))
    ds = ds.batch(batch_size)
    ds = ds.repeat()
    x, y =  ds.make_one_shot_iterator().get_next()
    return x,y

def train():
    max_grad_norm = config.get("max_grad_norm")
    learning_rate = config.get("learning_rate")
    vocabulary_size = config.get("vocabulary_size")
    num_hidden = config.get("num_hidden")
    layers = config.get("layers")
    num_outputs = config.get("num_outputs")
    batch_size = config.get("batch_size")
    time_steps = config.get("time_steps")

    x, y = next_data_batch()
    inputs, targets = tf.unstack(x, time_steps, 1), tf.unstack(y, time_steps, 1)  # Convert tensors to lists



    #tf.summary.tensor_summary("input[0]",inputs[0])
    inputs_encoded = [tf.one_hot(elem, depth=vocabulary_size) for elem in inputs]

    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(num_units=num_hidden, forget_bias=1.0) for _ in range(layers)],state_is_tuple=True)

    state = rnn_cell.zero_state(batch_size, tf.float32)

    outputs, after_state = rnn.static_rnn(rnn_cell, inputs_encoded, initial_state=state)

    flat_outputs = tf.reshape(tf.concat(axis=1, values=outputs), [-1, num_hidden])

    softmax_w = tf.Variable(tf.random_normal([num_hidden, num_outputs]))
    softmax_b = tf.Variable(tf.random_normal([vocabulary_size]))
    logits = tf.matmul(flat_outputs, softmax_w) + softmax_b

    flat_targets = tf.reshape(tf.concat(axis=0, values=targets), [-1])

    mean_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=flat_targets))
    tf.summary.scalar('mean_loss', mean_loss)

    trainable_vars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(mean_loss, trainable_vars), max_grad_norm)

    optimizer = tf.train.AdagradOptimizer(learning_rate)
    global_step = tf.Variable(0, trainable=False)

    train_op = optimizer.apply_gradients(zip(grads, trainable_vars), global_step=global_step)
    init_op = tf.global_variables_initializer()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tb_merged = tf.summary.merge_all()
        tb_writer = tf.summary.FileWriter('/tmp/train',
                                          sess.graph)
        sess.run(init_op)
        zero_state = rnn_cell.zero_state(batch_size, tf.float32)
        cur_state = sess.run(zero_state)
        loss_avg=0;
        save_freq = config.get('save_every_n_batches')
        vocabulary = functions.load_vocabulary()

        for i in range(1,100000000):
            #_, loss, cur_state, input0, output0 = sess.run([train_op, mean_loss, after_state, x, tf.nn.top_k(tf.nn.softmax(logits)).indices], feed_dict={state:cur_state})
            _, loss, cur_state = sess.run(
                [train_op, mean_loss, after_state],feed_dict={state: cur_state})

            #tb_writer.add_summary(tb_summary, i)
            loss_avg += loss
            if i%save_freq == 0:
                chars_processed = i*batch_size*time_steps
                epoch = float(chars_processed / config.get("epoch_len"))
                print("%f epoch; %d chars: Avg loss is %f"%(epoch, chars_processed,loss_avg/save_freq))
                save_path = saver.save(sess, config.get('model_save_path') % (loss_avg/save_freq))
                print("Model saved: %s" % save_path)
                #print(decode(vocabulary, input0[0]))
                #print(decode(vocabulary, output0.flatten()))
                loss_avg = 0

def decode(voc, data):
    res =[]
    for idx in data:
        #print(idx)
        res.append(voc[idx])
    return str(len(res)) + ":" + "".join(res)

if __name__ == '__main__':
    global vocabulary, data, encoded_data, char_to_idx
    if len(sys.argv) == 2:
        config_file = sys.argv[1]
        config.load(config_file)
    else:
        print("""
        Desired usage: runner.py config.json
        Right now we initialize with defaults and save them to train.json
        """)

        config.set('time_steps',200)
        config.set('layers',3)
        config.set('num_hidden',256)
        config.set('learning_rate', 0.07)
        config.set('num_input', 65)
        config.set('num_outputs', 65)
        config.set('batch_size', 1)
        config.set('input_file', "input.txt")
        config.set('vocabulary_file', "/tmp/voc.txt")
        config.set('vocabulary_size', 65)
        config.set('save_every_n_batches', 500)
        config.set('max_grad_norm',5.0)
        config.set('model_save_path',"/tmp/model_%f.ckpt")
        config.set('sample_length', 2000)

        config.dump("train.json")

    functions.generate_vocabulary()
    vocabulary = functions.load_vocabulary()
    data = open(config.get('input_file'), 'r').read()
    config.set("epoch_len", len(data))
    char_to_idx = {vocabulary[i]: i for i in range(len(vocabulary))}
    encoded_data = [char_to_idx[data[i]] for i in range(len(data))]


    train()