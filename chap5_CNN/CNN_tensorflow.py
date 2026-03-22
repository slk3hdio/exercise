import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()


learning_rate = 1e-4
keep_prob_rate = 0.7
max_epoch = 2000


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1.0})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1.0})
    return result


def weight_variable(shape):
    initial = tf.compat.v1.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool2d(x, ksize=2, strides=2, padding="SAME")


class DataSplit:
    def __init__(self, images, labels, seed=0):
        self.images = images
        self.labels = labels
        self.num_examples = images.shape[0]
        self.rng = np.random.default_rng(seed)
        self._index = 0
        self._shuffle()

    def _shuffle(self):
        indices = self.rng.permutation(self.num_examples)
        self.images = self.images[indices]
        self.labels = self.labels[indices]

    def next_batch(self, batch_size):
        if self._index + batch_size > self.num_examples:
            self._shuffle()
            self._index = 0

        start = self._index
        end = start + batch_size
        self._index = end
        return self.images[start:end], self.labels[start:end]


class MNISTData:
    def __init__(self):
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        train_images = train_images.reshape(-1, 784).astype(np.float32)
        test_images = test_images.reshape(-1, 784).astype(np.float32)
        train_labels = tf.keras.utils.to_categorical(train_labels, 10)
        test_labels = tf.keras.utils.to_categorical(test_labels, 10)

        self.train = DataSplit(train_images, train_labels, seed=0)
        self.test = DataSplit(test_images, test_labels, seed=1)


# define placeholder for inputs to network
xs = tf.compat.v1.placeholder(tf.float32, [None, 784]) / 255.0
ys = tf.compat.v1.placeholder(tf.float32, [None, 10])
keep_prob = tf.compat.v1.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])

# conv1 layer
W_conv1 = weight_variable([7, 7, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# conv2 layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# fc1 layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.compat.v1.nn.dropout(h_fc1, keep_prob=keep_prob)

# fc2 layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
prediction = tf.nn.softmax(logits)

# cross entropy
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=logits)
)
train_step = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(cross_entropy)


mnist = MNISTData()
with tf.compat.v1.Session() as sess:
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    for i in range(max_epoch):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: keep_prob_rate})
        if i % 100 == 0:
            print(compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))
