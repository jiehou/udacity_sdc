import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from project3.helper_functions import *
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# saved pickles #
train_filename = "./data/train.p"
test_filename = "./data/test.p"
valid_filename = "./data/valid.p"
train = load_pickle(train_filename)
test = load_pickle(test_filename)
valid = load_pickle(valid_filename)
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


"""
plt.subplot(1, 2, 1)
plt.hist(y_train, bins=43)
plt.title("histogram")
plt.subplot(1, 2, 2)
labels, dist_train = get_class_distribution(y_train, 43)
plt.bar(labels, dist_train)
plt.title("my historgram")
"""


img = np.squeeze(X_train[1520])
trans_img = np.squeeze(random_scale(img))
gray_img = np.squeeze(rgb_to_grayscale(img))
norm_img = np.squeeze(normalization(gray_img))
plot_two_images(img, norm_img, "original", "normalized grayscaled", True)
plt.show()

"""
class_labels, dist_train = get_class_distribution(y_train, 43)
X_fake, y_fake = generate_fake_data(X_train, y_train, dist_train)
print("#[I] X_fake.shape: ", X_fake.shape)
print("#[I] y_fake.shape: ", y_fake.shape)
_, dist = get_class_distribution(y_fake, 43)
plt.bar(class_labels, dist)
plt.title("historgram")
"""

"""
dict_traffic_sign_names = traffic_sign_names("./signnames.csv")
print("#[I] dict_traffic_sign_names[10]: ", dict_traffic_sign_names[str(10)])

test_images_set = [['./test/priority_road_12.jpg', 12],
                   ['./test/yield_13.jpg', 13],
                   ['./test/stop_14.jpg', 14],
                   ['./test/no_vehicles_15.jpg', 15],
                   ['./test/no_entry_17.jpg', 17],
                   ['./test/turn_right_ahead_33.jpg', 33],
                   ]
n_tests = len(test_images_set)
X_test_new = np.zeros([n_tests, 32, 32, 3], dtype=np.float32)
y_test_new = np.zeros([n_tests], dtype=np.int32)

for idx, item in enumerate(test_images_set):
    img = load_img(item[0])
    img = cv2.resize(img, (32, 32))
    X_test_new[idx, :, :, :] = img
    y_test_new[idx] = item[1]
    plt.subplot(2, 3, idx + 1)
    plt.imshow(img)
    plt.title(dict_traffic_sign_names[str(item[1])])
    plt.axis("off")
plt.show()
print(y_test_new)
X_test_new_processed = preprocess_dataset(X_test_new)
idx = 0
for img in X_test_new_processed:
    plt.subplot(2, 3, idx + 1)
    plt.imshow(np.squeeze(img), cmap="gray")
    plt.axis("off")
    idx = idx + 1
plt.show()
"""