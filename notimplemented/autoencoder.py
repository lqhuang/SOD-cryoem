import os
import shutil
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# import matplotlib.pyplot as plt

# -------------------------- util ----------------------------------- #
def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)

def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    # print("random_start_index=" + str(start_index), "batch_size=" + str(batch_size))
    return data[start_index:(start_index + batch_size)]

# weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Convolution and Pooling
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

# NN
def nn_layer(input_tensor, input_shape, output_shape, layer_name,
             cnn=False, act=tf.nn.relu, pooling=False):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            W = self.weight_variable(input_shape)
        with tf.name_scope('biases'):
            b = self.bias_variable(output_shape)
        if cnn:
            with tf.name_scope('convolution'):
                h = act(conv2d(input_tensor, W) + b)
        else:
            with tf.name_scope('activation'):
                h = act(tf.matmul(input_tensor, W) + b)
    if pooling:
        with tf.name_scope('pooling_layer'):
            # h_pool = max_pool_2x2(h)
            h_pool = avg_pool_2x2(h)
        return h_pool, W, b
    else:
        return h, W, b

# -------------------- autoencoder --------------------------------- #
class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self, n_input, n_hidden,
                 transfer_function=tf.nn.softplus,
                 act_function=tf.nn.relu,
                 optimizer=tf.train.AdamOptimizer(), scale=0.1,
                 two_hidden_layers=False):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function

        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale

        with tf.name_scope('Autoencoder'):
            with tf.name_scope('Input_Layer'):
                self.x = tf.placeholder(tf.float32, [None, self.n_input])
                image_original = tf.reshape(self.x, [-1, 28, 28, 1])
                tf.summary.image('original-input', image_original, 4)
                self.input = self.x + scale * tf.random_normal((n_input,))
                image_shaped_input = tf.reshape(self.input, [-1, 28, 28, 1])
                tf.summary.image('autoencoder-input', image_shaped_input, 4)

            with tf.name_scope('Layer_1'):
                with tf.name_scope('Weights'):
                    W1 = tf.Variable(xavier_init(self.n_input, self.n_hidden))
                with tf.name_scope('Biases'):
                    b1 = bias_variable([self.n_hidden])
                with tf.name_scope('Wx_plus_b'):
                    layer1 = tf.add(tf.matmul(self.input, W1), b1)
                    self.hidden_1 = self.transfer(layer1)

            if two_hidden_layers:
                with tf.name_scope('Layer_2'):
                    with tf.name_scope('Weights'):
                        W2 = weight_variable([self.n_hidden, self.n_hidden])
                    with tf.name_scope('Biases'):
                        b2 = bias_variable([self.n_hidden])
                    with tf.name_scope('Wx_plus_b'):
                        layer2 = tf.add(tf.matmul(self.hidden_1, W2), b2)
                        self.hidden_2 = self.transfer(layer2)

            with tf.name_scope('reconstruction'):
                with tf.name_scope('Weights'):
                    W_reco = weight_variable([self.n_hidden, self.n_input])
                with tf.name_scope('Biases'):
                    b_reco = bias_variable([self.n_input])
                with tf.name_scope('Wx_plus_b'):
                    if two_hidden_layers:
                        self.reconstruction = tf.add(tf.matmul(self.hidden_2, W_reco), b_reco)
                    else:
                        self.reconstruction = tf.add(tf.matmul(self.hidden_1, W_reco), b_reco)
                image_shaped_output = tf.reshape(self.input, [-1, 28, 28, 1])
                tf.summary.image('autoencoder-output', image_shaped_input, 4)

        # loss function
        with tf.name_scope('loss_function'):
            with tf.name_scope('cost'):
                self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
                tf.summary.scalar('cost', self.cost)
        with tf.name_scope('Optimizer'):
            self.optimizer = optimizer.minimize(self.cost)


        self.merged = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.train_writer = tf.summary.FileWriter(os.path.join(log_dir, 'autoencoder_train'), self.sess.graph)
        self.test_writer = tf.summary.FileWriter(os.path.join(log_dir, 'autoencoder_test'))

    def partial_fit(self, X):
        opt = self.sess.run(self.optimizer,
                            feed_dict={self.x: X, self.scale: self.training_scale})
        # return cost
    def calc_train_cost(self, X):
        train_summary, train_cost = self.sess.run([self.merged, self.cost],
                                                  feed_dict={self.x: X, self.scale: self.training_scale})
        return train_summary, train_cost
    def calc_total_cost(self, X):
        test_summary, test_cost = self.sess.run([self.merged, self.cost],
                                                feed_dict={self.x: X, self.scale: self.training_scale})
        return test_summary, test_cost

    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X, self.scale: self.training_scale})

    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction,
                             feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction,
                             feed_dict={self.x: X, self.scale: self.training_scale})

    def close(self):
        self.test_writer.close()
        self.train_writer.close()

# run
log_dir = '/tmp/tensorflow/mnist/logs/autoencoder_noise_scale_0_1'
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

n_samples = int(mnist.train.num_examples)
training_epochs = 40
batch_size = 256
display_step = 1
autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=784,
                                               n_hidden=200,
                                               transfer_function=tf.nn.softplus,
                                               optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                               scale=0.1, two_hidden_layers=False)

total_batch = int(n_samples / batch_size)
for epoch in range(training_epochs):
    avg_cost = 0
    print("total_batch:"+str(total_batch))
    for i in range(total_batch):
        step = epoch * total_batch + i
        batch_xs = get_random_block_from_data(X_train, batch_size)
        autoencoder.partial_fit(batch_xs)
        train_summary, train_cost = autoencoder.calc_train_cost(batch_xs)
        autoencoder.train_writer.add_summary(train_summary, step)
    if epoch % display_step == 0:
        step = epoch * total_batch
        test_summary, test_cost = autoencoder.calc_total_cost(X_test)
        autoencoder.test_writer.add_summary(test_summary, step)
        print("Epoch", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(test_cost))

test_summary, total_cost = autoencoder.calc_total_cost(X_test)
autoencoder.test_writer.add_summary(test_summary, training_epochs * total_batch + total_batch - 1)
print("Total cost:", total_cost)

autoencoder.close()