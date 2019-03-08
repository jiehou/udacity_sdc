import math
import numpy as np
import csv
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Cropping2D


OFFSET = 0.2
OFFSETS = [0.0, OFFSET, -OFFSET]  # center, left and right images
INPUT_SHAPE = (160, 320, 3)


def random_horizontal_flip(x, y):
    flip = np.random.rand()
    if flip <= 0.5:
        x = cv2.flip(x, 1)
        y = -1.0 * y
    return x, y


def get_samples(csv_filepath):
    samples = []
    with open(csv_filepath) as infile:
        reader = csv.reader(infile)
        for sample in reader:
            samples.append(sample)
    return samples[1:]  # remove the header (center,left,right,steering,throttle,brake,speed)


def generator(samples, is_training, base_path="./data/", batch_size=64):
    """
    generate a batch of images from the saved log file
    :param samples: sample lines from the saved log file
    :param base_path: directory of the saved images and log file
    :param batch_size: batch size
    """
    n_samples = len(samples)
    n_views = 3  # center, left, right (in order to make training set more generalized)
    while True:
        # declare output data
        x_out = []
        y_out = []
        # fill batch
        for i in range(0, batch_size):
            # get random idx in the data set
            idx = np.random.randint(n_samples)
            sample = samples[idx]
            center_steering = float(sample[3])
            if is_training:
                idx_view = np.random.randint(n_views)
                # read image and steering angle
                file_name = str(sample[idx_view]).split("\\")[-1]
                img_filepath = base_path + "IMG/" + file_name.strip()  # "./data/IMG/center_2016_12_01_13_32_39_212.jpg"
                # print("#[D]: ", img_filepath)
                x_i = cv2.imread(img_filepath)
                y_i = center_steering + OFFSETS[idx_view]
                # data augmentation (flip)
                x_i, y_i = random_horizontal_flip(x_i, y_i)
            else:
                # validation (only consider center cases)
                x_i = cv2.imread(base_path + "IMG/" + str(sample[0]).split("\\")[-1])  # center
                y_i = center_steering
            # add to batch
            x_out.append(x_i)
            y_out.append(y_i)
        yield (np.array(x_out), np.array(y_out))


def build_model():
    """
    nvidia end-to-end self-driving-car cnn
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/255 - 0.5, input_shape=INPUT_SHAPE))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    # five convolutional layers
    model.add(Conv2D(24, (5, 5), activation="relu", strides=2))
    model.add(Conv2D(36, (5, 5), activation="relu", strides=2))
    model.add(Conv2D(48, (5, 5), activation="relu", strides=2))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    # dropout
    model.add(Dropout(0.7))  # keep proability
    model.add(Flatten())
    # fully connected layers
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.summary()
    print("#[I] model is ready")
    return model


def train_model(model, train_samples, validation_samples, batch_size):
    """
    train the model
    :param model: built-model
    :param X_train: train samples
    :param y_train: train targets
    :param X_valid: validation samples
    :param y_valid: validation targets
    """
    checkpoint = ModelCheckpoint("./tmp/model_{epoch:03d}.hdf5",
                                 monitor="val_loss",
                                 verbose=0,
                                 save_best_only=False,
                                 mode="auto")

    train_generator = generator(train_samples, True, batch_size=batch_size)
    validation_generator = generator(validation_samples, False, batch_size=batch_size)
    n_train_samples = len(train_samples)
    n_validation_samples = len(validation_samples)

    model.compile(loss="mse", optimizer=Adam(lr=0.0001))
    model.fit_generator(train_generator,
                        steps_per_epoch=math.ceil(n_train_samples / batch_size) * 5,
                        validation_data=validation_generator,
                        validation_steps=math.ceil(n_validation_samples / batch_size),
                        epochs=16,
                        callbacks=[checkpoint],
                        verbose=1)
    return model


def save_model(model):
    print("#[I] saving model")
    model_json = model.to_json()
    # save model
    with open("model.json", "w+") as out_file:
        out_file.write(model_json)
    # save weights
    model.save("model.h5")
    print("#[I] saved model to disk")


def main():
    # 1) get samples from the saved log file
    csv_filepath = "./data/driving_log.csv"
    samples = get_samples(csv_filepath)
    print("#[I] len(samples): ", len(samples))
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    print("#[I] len(train_samples): ", len(train_samples))
    print("#[I] len(validation_samples): ", len(validation_samples))
    # 2) build model
    model = build_model()
    # 3) train model
    batch_size = 64
    model = train_model(model, train_samples, validation_samples, batch_size)
    # 4) save model
    save_model(model)


if __name__ == "__main__":
    main()