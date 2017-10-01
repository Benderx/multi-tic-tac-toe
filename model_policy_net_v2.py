import tensorflow as tf
import itertools
import os
import h5py
import random
import numpy as np
import math
from threading import Timer
import time


# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 3
display_step = 1


# Network Parameters
n_hidden_1 = 512 # 1st layer number of features
n_hidden_2 = 512 # 2nd layer number of features
n_input = 18 # MNIST data input (img shape: 28*28)
n_classes = 9 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])



# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
# initialized via http://cs231n.github.io/neural-networks-2/

# print(math.sqrt(2.0/(n_input)))
# exit()
h1_weight = np.random.randn(n_input, n_hidden_1) * math.sqrt(2.0/(n_input))
h2_weight = np.random.randn(n_hidden_1, n_hidden_2) * math.sqrt(2.0/(n_hidden_1))
out_weight = np.random.randn(n_hidden_2, n_classes) * math.sqrt(2.0/(n_hidden_2))


weights = {
    'h1': tf.Variable(h1_weight, dtype=np.float32),
    'h2': tf.Variable(h2_weight, dtype=np.float32),
    'out': tf.Variable(out_weight, dtype=np.float32)
}
biases = {
    'b1': tf.Variable(tf.constant(.01, shape=(n_hidden_1,))),
    'b2': tf.Variable(tf.constant(.01, shape=(n_hidden_2,))),
    'out': tf.Variable(tf.constant(.01, shape=(n_classes,)))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)


# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# saver
saver = tf.train.Saver()




def random_state_gen(f, random_sample):
    for move_id in random_sample:
        state = f['random_games'][move_id]
        winner = 1
        if state[1] == 2:
            winner = 2

        game_len = state[2]

        if winner == 1:
            yield state[12:21], state[3:12], state[21:]
        else:
            yield state[3:12], state[12:21], state[21:]
    yield 'over', None, None



def get_last(f, def_shape):
    def_shape -= 1
    if f['random_games'][def_shape][0] != 0:
        return def_shape
    while def_shape >= 0:
        if f['random_games'][def_shape][0] != 0:
            return def_shape + 1
        def_shape -= 1



def get_batch(batch_size):
    size = 3
    path = os.path.join('data', 'random', str(size), 'mytestfile1.hdf5')

    with h5py.File(path, 'r') as f:
        moves_len = f['random_games'].shape[0]

        last_move_index = get_last(f, moves_len)
        print(last_move_index)


        games_range_gen = range(last_move_index+1)
        random_sample = random.sample(games_range_gen, last_move_index)

        print('training from {0} games'.format(last_move_index))

        state_gen = random_state_gen(f, random_sample)
        while True:
            i = 0
            xarr = []
            yarr = []

            while i < batch_size:
                x1ret, x2ret, yret = next(state_gen)
                # print('x1ret', x1ret)
                if x1ret == 'over':
                    print('finished {0} states, this batch only has {1}/{2} states'.format(last_move_index+1, i, batch_size))
                    state_gen = random_state_gen(f, random_sample)
                    break

                xarr.append(np.concatenate((x1ret, x2ret)))
                yarr.append(yret)
                i += 1

            if len(xarr) == 0:
                continue
            yield xarr, yarr



all_models_path = os.path.join('models', '{0}'.format('policy_net_v2'))
models = os.listdir(all_models_path)
model_num = str(len(models))
print(model_num)
model_path = os.path.join(all_models_path, '{0}'.format(str(model_num)))


time_seconds = 180
global lmao; lmao = True
def training_time_up():
    global lmao;
    print('Training period over for: {0}'.format(model_path))
    print('Training ran for {0} seconds'.format(time_seconds))
    lmao = False
    exit()


# duration is in seconds
t = Timer(time_seconds, training_time_up)
t.start()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    batch_gen = get_batch(batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        avg_sum = 0.
        avg_num = 0
        batches = 4000000

        # Loop over all batches
        for i in range(batches):
            batch_x, batch_y = next(batch_gen)

            # print('\n, STARTING NEW \n')
            # Run optimization op (backprop) and cost op (to get loss value)
            op, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            # print('ok')
            # print(pred)
            # print('lol')
            # print(c)

            avg_sum += c
            avg_num += 1
            if not lmao:
                break
        if not lmao:
            break

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_sum / avg_num))

    print("final cost=", "{:.9f}".format(avg_sum / avg_num))
    print("Optimization Finished!")

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    save_path = os.path.join(model_path, 'ITSYABOI')
    saver.save(sess, save_path)
    print("Model saved in file: {0}".format(save_path))

    # # Test model
    # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # # Calculate accuracy
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))