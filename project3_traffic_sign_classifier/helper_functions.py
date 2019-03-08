import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import csv
from tensorflow.contrib.layers import flatten


# plot two images in one row
def plot_two_images(img1, img2, img1_title="", img2_title="", gray=True):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    if gray:
        ax1.imshow(img1, cmap="gray")
    else:
        ax1.imshow(img1)
    ax1.set_title(img1_title)
    if gray:
        ax2.imshow(img2, cmap="gray")
    else:
        ax2.imshow(img2)
    ax2.set_title(img2_title)


def load_pickle(filename):
    with open(filename, mode="rb") as f:
        saved_pickle = pickle.load(f)
        return saved_pickle


def get_class_distribution(y, n_classes):
    """
    :param y: labels
    :param n_classes: total number of classes
    :return: class distribution, in other words, which class has how many numbers of examples
    """
    labels = np.array(range(0, n_classes))
    n_examples = np.zeros(n_classes, dtype=np.int32)
    for i in labels:
        n_examples[i] = (y == i).sum()  # y is np.array()
    return labels, n_examples


# generate fake data #
def apply_affine_transform(img, transform):
    height, width = img.shape[0:2]
    return cv2.warpAffine(img, transform, (width, height))


def random_translate(img, max_relative_displacement=0.10):
    height, weight = img.shape[0:2]
    tx = np.random.uniform(-max_relative_displacement * weight, max_relative_displacement * weight)
    ty = np.random.uniform(-max_relative_displacement * height, max_relative_displacement * height)
    transform = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float32)
    return apply_affine_transform(img, transform)


def random_rotate(img, max_angel_degree=5.0):
    theta = np.random.uniform(-max_angel_degree, max_angel_degree) * np.pi / 180.0
    c = np.cos(theta)
    s = np.sin(theta)
    transform = np.array([[c, -s, 0.0], [s, c, 0.0]], dtype=np.float32)
    return apply_affine_transform(img, transform)


def random_scale(img, max_scale_factor=0.05):
    sx = 1.0 + np.random.uniform(-max_scale_factor, max_scale_factor)
    sy = 1.0 + np.random.uniform(-max_scale_factor, max_scale_factor)
    transform = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0]], dtype=np.float32)
    return apply_affine_transform(img, transform)


def jitter_image(img):
    fn = np.random.randint(0, 3)
    if fn == 0:
        return random_translate(img)
    elif fn == 1:
        return random_rotate(img)
    else:
        return random_scale(img)


def generate_fake_data(X, y, old_class_distribution, extend_factor=10):
    X_fake = []
    y_fake = []
    # get final size of datasets
    final_size = extend_factor * X.shape[0]
    n_classes = len(old_class_distribution)
    n_examples_class = int(np.ceil(final_size / n_classes))
    n_fakes_class = np.ceil((n_examples_class - old_class_distribution) / old_class_distribution).astype(int)

    for i in range(X.shape[0]):
        class_idx = y[i]
        # generate fake images
        for j in range(n_fakes_class[class_idx]):
            X_fake.append(jitter_image(X[i]))
            y_fake.append(class_idx)
    X = np.concatenate((X, X_fake), axis=0)
    y = np.concatenate((y, y_fake), axis=0)
    return X, y


def rgb_to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# zero mean and unit variance
def normalization(x):
    a = 0.1
    b = 0.9
    x_min = 0
    return a + (x - x_min) * (b - a) / 255


def simple_normalization(x):
    return (x - 128) / 128


def preprocess_dataset(dataset):
    shape = np.array(dataset.shape)  # [34799, 32, 32, 3]
    shape[3] = 1  # grayscaled image
    result_dataset = np.zeros(shape)
    # print('#[D] result_dataset.shape: ', result_dataset.shape)
    for i in range(shape[0]):
        result_dataset[i, :, :, :] = np.expand_dims(rgb_to_grayscale(dataset[i]), axis=2)
    result_dataset = normalization(result_dataset)
    return result_dataset


# cnn #
def weights_variable(shape, mu, sigma):
    """
    create weights
    :param shape: filter [height, width, input_depth, output_depth]
    :param mu: mean
    :param sigma: stddev
    :return: create weights
    """
    return tf.Variable(tf.truncated_normal(shape, mean=mu, stddev=sigma))


def biases_variable(shape, mu=0.05):
    """
    create biases
    :param shape: [output_depth]
    :param mu: mean
    :return: created biases
    """
    return tf.Variable(tf.zeros(shape) + mu)


def conv2d(x, W, b, stride=1, conv_padding="VALID"):
    """
    create convolutional layer, and the used activation is relu.
    :param x:
    :param W:
    :param b:
    :param stride:
    :return:
    """
    x = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=conv_padding)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2, str_padding="VALID"):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding=str_padding)


def LeNet_v2(x, dropout_keep_prob):
    mu = 0
    sigma = 0.1
    # layer 1: convolutiona layer 32x32x1 -> 28x28x16
    # max pooling: 28x28x16 -> 14x14x16

    # layer 2: convolutional layer 14x14x16 -> 14x14x32 ("SAME")
    # dropout: 0.85

    # layer 3: convolutional layer 14x14x16 -> 10x10x64
    # max pooling: 10x10x64 -> 5x5x64

    # flatten
    # layer 3: fully connected 1600 -> 800
    # layer 4: fully connected 800 -> 200
    # layer 5: fully connected 200 -> 43
    weights = {
        'c1': weights_variable([5, 5, 1, 16], mu, sigma),
        'c2': weights_variable([5, 5, 16, 32], mu, sigma),
        'c3': weights_variable([5, 5, 32, 64], mu, sigma),
        'fc1': weights_variable([1600, 800], mu, sigma),
        'fc2': weights_variable([800, 200], mu, sigma),
        'fc3': weights_variable([200, 43], mu, sigma)
    }
    biases = {
        'c1': biases_variable([16]),
        'c2': biases_variable([32]),
        'c3': biases_variable([64]),
        'fc1': biases_variable([800]),
        'fc2': biases_variable([200]),
        'fc3': biases_variable([43])
    }
    # layer 1: convolutional layer #
    # 32x32x1 -> 28x28x16
    conv1 = conv2d(x, weights['c1'], biases['c1'], stride=1)  # VALID
    # max pooling: 28x28x16 -> 14x14x16
    conv1 = maxpool2d(conv1, 2)

    # layer 2: convolutional layer #
    # 14x14x16 -> 14x14x32
    conv2 = conv2d(conv1, weights['c2'], biases['c2'], stride=1, conv_padding="SAME")
    conv2 = tf.nn.dropout(conv2, keep_prob=dropout_keep_prob)

    # layer 3: convolutional layer #
    # 14x14x32 -> 10x10x64
    conv3 = conv2d(conv2, weights['c3'], biases['c3'], stride=1)  # VALID
    # max pooling: 10x10x64 -> 5x5x64
    conv3 = maxpool2d(conv3, 2)

    fc0 = flatten(conv3)
    # layer 3: fully connected layer #
    # 400 -> 120
    fc1 = tf.matmul(fc0, weights['fc1']) + biases['fc1']
    # relu activation
    fc1 = tf.nn.relu(fc1)
    # layer 4: fully connected layer #
    # 120 -> 84
    fc2 = tf.matmul(fc1, weights['fc2']) + biases['fc2']
    # relu activation
    fc2 = tf.nn.relu(fc2)
    # layer 5: fully connected layer #
    logits = tf.matmul(fc2, weights['fc3']) + biases['fc3']
    return logits


def LeNet_v1(x):
    mu = 0
    sigma = 0.1
    # layer 1: convolutiona layer 32x32x1 -> 28x28x6
    # max pooling: 28x28x6 -> 14x14x6
    # layer 2: convolutional layer 14x14x6 -> 10x10x16
    # max pooling: 10x10x16 -> 5x5x16
    # flatten
    # layer 3: fully connected 400 -> 120
    # layer 4: fully connected 120 -> 84
    # layer 5: fully connected 84 -> 43
    weights = {
        'c1': weights_variable([5, 5, 1, 6], mu, sigma),
        'c2': weights_variable([5, 5, 6, 16], mu, sigma),
        'fc1': weights_variable([400, 120], mu, sigma),
        'fc2': weights_variable([120, 84], mu, sigma),
        'fc3': weights_variable([84, 43], mu, sigma)  # 43 / 10
    }
    biases = {
        'c1': biases_variable([6]),
        'c2': biases_variable([16]),
        'fc1': biases_variable([120]),
        'fc2': biases_variable([84]),
        'fc3': biases_variable([43])  # 43 / 10
    }
    # layer 1: convolutional layer #
    # 32x32x1 -> 28x28x6
    conv1 = conv2d(x, weights['c1'], biases['c1'], stride=1)
    # max pooling: 28x28x6 -> 14x14x6
    conv1 = maxpool2d(conv1, 2)
    # layer 2: convolutional layer #
    # 14x14x6 -> 10x10x16
    conv2 = conv2d(conv1, weights['c2'], biases['c2'], stride=1)
    # max pooling: 10x10x16
    conv2 = maxpool2d(conv2, 2)
    fc0 = flatten(conv2)
    # layer 3: fully connected layer #
    # 400 -> 120
    fc1 = tf.matmul(fc0, weights['fc1']) + biases['fc1']
    # relu activation
    fc1 = tf.nn.relu(fc1)
    # layer 4: fully connected layer #
    # 120 -> 84
    fc2 = tf.matmul(fc1, weights['fc2']) + biases['fc2']
    # relu activation
    fc2 = tf.nn.relu(fc2)
    # layer 5: fully connected layer #
    logits = tf.matmul(fc2, weights['fc3']) + biases['fc3']
    return logits


def LeNet_v0(x):
    mu = 0
    sigma = 0.1
    # weigths and biases
    weights = {
        'wc1': tf.Variable(tf.truncated_normal([5, 5, 1, 6], mean=mu, stddev=sigma)),
        'wc2': tf.Variable(tf.truncated_normal([5, 5, 6, 16], mean=mu, stddev=sigma)),
        'fc1': tf.Variable(tf.truncated_normal([400, 120], mean=mu, stddev=sigma)),
        'fc2': tf.Variable(tf.truncated_normal([120, 84], mean=mu, stddev=sigma)),
        'fc3': tf.Variable(tf.truncated_normal([84, 10], mean=mu, stddev=sigma))
    }
    biases = {
        'bc1': tf.Variable(tf.zeros([6])),
        'bc2': tf.Variable(tf.zeros([16])),
        'bc3': tf.Variable(tf.zeros([120])),
        'bc4': tf.Variable(tf.zeros([84])),
        'bc5': tf.Variable(tf.zeros([10]))
    }

    # layer 1 #
    # convolutional input: 32x32x1 -> output: 28x28x6
    conv1 = conv2d(x, weights['wc1'], biases['bc1'], 1)
    # pooling 28x28x6 -> 14x14x6
    conv1 = maxpool2d(conv1, 2)
    # layer 2 #
    # convolutional input: 14x14x6 ->10x10x16
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], 1)  # stride 1
    # pooling 10x10x16 -> 5x5x16
    conv2 = maxpool2d(conv2, 2)
    # flatten #
    fc0 = flatten(conv2)
    # layer 3 #
    # 400 -> 120
    fc1 = tf.add(tf.matmul(fc0, weights['fc1']), biases['bc3'])
    fc1 = tf.nn.relu(fc1)
    # layer 4 #
    # 120 -> 84
    fc2 = tf.add(tf.matmul(fc1, weights['fc2']), biases['bc4'])
    fc2 = tf.nn.relu(fc2)
    # layer 5 #
    # 84 -> 10
    fc3 = tf.add(tf.matmul(fc2, weights['fc3']), biases['bc5'])
    return fc3


def generate_batches(batch_size, features, labels):
    assert(len(features) == len(labels))
    output_batches = []
    num_examples = len(features)
    for start_i in range(0, num_examples, batch_size):
        end_i = start_i + batch_size
        batch = [features[start_i:end_i], labels[start_i:end_i]]
        output_batches.append(batch)
    return output_batches


def load_img(img_path):
    return mpimg.imread(img_path)


def traffic_sign_names(csv_file_path):
    dict_sign_names = dict()
    with open(csv_file_path, 'r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            sign_id = row[0]
            sign_name = row[1]
            dict_sign_names[sign_id] = sign_name
    return dict_sign_names


