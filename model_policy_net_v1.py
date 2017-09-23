import tensorflow as tf
import itertools
import os
import h5py
import random
import numpy as np

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
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
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
    for game_id in random_sample:
        upper = 4
        winner = 1
        game = f['random_games'][game_id]

        # print(game)

        if game[0][1] == 2:
            winner = 2

        game_len = game[0][2]//2

        # print(game_len)

        if winner == 1:
            while True:
                r = random.randint(0, game_len)
                state = game[r*2]
                yield state[12:21], state[3:12], state[21:]
                break
        else:
            while True:
                r = random.randint(0, game_len-1)
                state = game[r*2 + 1]
                yield state[3:12], state[12:21], state[21:]
                break
    yield 'over', None, None


def get_batch(batch_size):
    size = 3
    path = os.path.join('data', 'random', '3', 'mytestfile.hdf5')

    with h5py.File(path, 'r') as f:
        games_len = f['random_games'].shape[0]

        games_range_gen = range(games_len)
        random_sample = random.sample(games_range_gen, games_len)

        print('training from {0} games'.format(games_len))

        state_gen = random_state_gen(f, random_sample)
        while True:
            i = 0
            xarr = []
            yarr = []

            while i < batch_size:
                x1ret, x2ret, yret = next(state_gen)
                # print('x1ret', x1ret)
                if x1ret == 'over':
                    print('finished {0} states, this batch only has {1}/{2} states'.format(games_len, i, batch_size))
                    state_gen = random_state_gen(f, random_sample)
                    break

                xarr.append(np.concatenate((x1ret, x2ret)))
                yarr.append(yret)
                i += 1

            if len(xarr) == 0:
                continue
            yield xarr, yarr







# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    batch_gen = get_batch(batch_size)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        batches = 50000

        # Loop over all batches
        for i in range(batches):
            batch_x, batch_y = next(batch_gen)

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / batches

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    save_path = os.path.join('models', 'policy_net_v1')
    saver.save(sess, save_path)
    print("Model saved in file: {0}".format(save_path))

    # # Test model
    # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # # Calculate accuracy
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))