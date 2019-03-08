# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/model_summary.png "ModelSummary"
[image2]: ./output_images/example_flipped_image.png "FlippedImage"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

### Files Submitted & Code Quality
---
#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode
My project includes the following files:
* model.py containing the script to create and train the model
* model.h5 containing a trained convolution neural network 
* project4_writeup.md summarizing the results
* drive.py for driving the car in autonomous mode (it is from Udacity, and I do not change anything here)

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable
The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture
---
Our network is based on the NVIDIA model, which has been published by NVIDIA and used for the end-to-end self-driving test. It is a convolutional neural network, which works well with supervised image regression problems. It maps images captured by a front-facing camera directly to steering commands. In this project, we have the same problem as solved by the NVIDIA. This is the reason why we choose its published model.

#### 1. An appropriate model architecture has been employed
Compared to the original NVIDIA model, I have added the following modifications:
* I added a Cropping2D layer after normalization. The cameras (center, left and right) in the simulator capture images that are of size 160x320. Not all of these pixels contain useful information. The top portion of the images captures some background information such as hills or sky. The bottom portion of the image captures the hood of the car.
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

* I added an additional dropout layer to avoid overfitting after the five convolutional layers. The keep probability is set as 0.7.

In summary, our network consists of the following layers:
* Input: the image of size (160, 320, 3)
* Normalization layer: the range is [-0.5, 0.5]. It is achieved by the Keras Lamda layer that provides a convenient way to parallelize image normalization.
* Cropping layer: it cropped 70 pixels from the top and 25 pixels from the bottom of the input images.
* Five convolutional layers: they were designed to perform feature extraction. In each of the first three convolutional layers, a strided convolution with a 5x5 kernel size 5x5 and a (2, 2) stride is used. A non-strided convolution with a 3x3 kernel size is used in each of the last two convolutional layers.
* Dropout layer: it was introduced to prevent overfitting. The keep probability is set as 0.7.
* Three fully connected layers: a steering control value was output.

The model summary is displayed as follows:

![Model summary][image1]

As can be seen, there are totally 348,219 parameters in our model.
```python
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
    model.add(Dropout(0.7))
    model.add(Flatten())
    # fully-connected layers
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.summary()
    print("#[I] model is ready")
    return model
```

### Training strategy
---
#### 1. Model parameter tuning
The model used an adam optimizer, the learning rate is set as 0.0001. Because first I tried the default value 0.001, the difference between accuracies from training and validatin sets is large. Then, I set the learning rate as 0.0001, I found the difference become smaller. Morever, the trained model works well on the simulator.

#### 2. Appropriate training data
Training data was collected on our local computer that runs Windows 10 on it. Initially, I tried to drive the car only in the middle of the lane during the training mode. I collected data from three laps. After training, I found that the car runs against the guard bar on the bridge under the autonomous mode.

Based on suggestions and materials from Udacity, I should collect more data. Finally, five laps of data were collected. The first two laps are normal laps. In order to teach the car what to do when it's off on the side of the road. In the third lap, I wander off to the side of the road and steer back to the center constantly. In the last two laps, I drive the car in the opposite direction. The images recorded in the counter-clockwise laps make our model generalize better.

Totally, there are 10580 lines in our **driving_log.csv** file. It means that there are **10580x3=31740** images.

#### 3. Training data generator
In order to make our model generalize better, I used the following augmentation technique together with Python generator to generate unlimited number of images. Generator functions allow us to declare a function that behaves like an iterator, which has the advantage of saving memory space when a large data set need to be processed.
* Randomly select center, left or right images
* For left image, steering angle is set as: the corresponding center steering angle + 0.2
* For right image, steering angle is equal to: the corresponding center steering angle - 0.2
* Randomly flip image with help of cv2.flip

Following is an example of image flipping:
![Example of image flipping][image2]

#### 4. Training, validation and test
I split the collected samples into train and validation set using following code:
```python
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
```
20% of our samples is split as the validation set.

I used the training set for training the model. The validation set helped to determine whether overfitting or underfitting happens. Following is our training configuration:

* batch size: 64
* number of epochs: 16
* steps_per_epoch: math.ceil(n_train_samples / batch_size) * 5. Because we want to take into account of images from left or right cameras. 
* validation_step: math.ceil(n_validation_samples / batch_size)

```python
model.compile(loss="mse", optimizer="adam")
model.fit_generator(train_generator,
                    steps_per_epoch=math.ceil(n_train_samples / batch_size) * 5,
                    validation_data=validation_generator,
                    validation_steps=math.ceil(n_validation_samples / batch_size),
                    epochs=16,
                    callbacks=[checkpoint],
                    verbose=1)
```

### Simulation output
---
The output video named **run1.mp4** can be found unter CarND-Behavioral-Cloning-P3. The test configuration is:
* Graphics: 960x720
* Graphics quality: Beautiful

### Summary
---
1. First, I trained our model on my own computer, and I wanted to test the trained model on it. I met the following error: GET /socket.io/?EIO=4&transport=websocket HTTP/1.1 404. Until now, I still do not have a solution to this problem.

2. I found that after training, under autonomous mode our car drives around 9 MPH. I have some doubts about that speed. Because in my opinion, speed influences the steering angle during the training mode. However, under the autonomous mode, the car drives at the same speed.

3. The collected data is very important for training our model. When a model does not work well, not only hyper-parameters but also training data need to be checked.