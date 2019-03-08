import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from project3.helper_functions import *
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# saved pickles #
train_filename = "./data/train.p"
test_filename = "./data/test.p"
valid_filename = "./data/valid.p"
train = load_pickle(train_filename)
test = load_pickle(test_filename)
valid = load_pickle(valid_filename)

# retrieve data #
features_str = "features"
labels_str = "labels"
X_train, y_train = train[features_str], train[labels_str]
X_valid, y_valid = valid[features_str], valid[labels_str]
X_test, y_test = test[features_str], test[labels_str]
assert(len(X_train) == len(y_train))
assert(len(X_valid) == len(y_valid))
assert(len(X_test) == len(y_test))
print('#[D] X_train.shape: ', X_train.shape)
print('#[D] y_train.shape: ', y_train.shape)

# data preprocessing #
X_train_processed = preprocess_dataset(X_train)
X_test_processed = preprocess_dataset(X_test)

# split into training, validation and test sets
X_train_split, X_test_split, y_train_split, y_test_split = \
    train_test_split(X_train_processed, y_train, test_size=0.2, random_state=1665)
print('#[I] size of X_train_split = ', X_train_split.shape)
print('#[I] size of X_test_split = ', X_test_split.shape)
print('#[I] size of y_train_split = ', y_train_split.shape)
print('#[I] size of y_valid_split = ', y_test_split.shape)

# network training #
n_epochs = 10
learning_rate = 0.001
batch_size = 128

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
dropout_keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, 43)

# logits = LeNet_v1(x)
logits = LeNet_v2(x, dropout_keep_prob=dropout_keep_prob)

# training pipeline
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_operation = optimizer.minimize(loss_operation)

# evaluation pipeline
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(batch_size, X_data, y_data):
    num_examples = len(X_data)
    # print("  #[D] evaluate num_examples: ", num_examples)
    eval_batches = generate_batches(batch_size, X_data, y_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for batch_features, batch_labels in eval_batches:
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_features, y: batch_labels, dropout_keep_prob: 1.0})
        # print("  #[D] evaluate len(batch_features): ", len(batch_features))
        # print("  #[D] accuracy: ", accuracy)
        total_accuracy += (accuracy * len(batch_features))
    return total_accuracy / num_examples


"""
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('# training...')
    for i in range(n_epochs):
        X_train_split, y_train_split = shuffle(X_train_split, y_train_split)
        train_batches = generate_batches(batch_size, X_train_split, y_train_split)
        for batch_features, batch_labels in train_batches:
            sess.run(training_operation, feed_dict={x: batch_features, y: batch_labels, dropout_keep_prob: 0.7})
        validation_accuracy = evaluate(batch_size, X_test_split, y_test_split)
        print("# epoch: ", i + 1)
        print("# validation accuracy = {:.3f}".format(validation_accuracy))
    saver.save(sess, "./lenet")
    print("# model saved")
"""

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph("./lenet_traffic_sign.meta")
    saver.restore(sess, "./lenet_traffic_sign")
    test_accuracy = evaluate(batch_size, X_test_processed, y_test)
    print("#[I] test_accuracy: ", test_accuracy)