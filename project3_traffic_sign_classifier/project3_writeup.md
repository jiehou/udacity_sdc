# **Traffic Sign Recognition** 

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images


[//]: # (Image References)

[image1]: ./output_images/1_visualize_classes.png "DifferentClasses"
[image2]: ./output_images/2_class_dist_training_set.png "ClassDist"
[image3]: ./output_images/3_example_of_iamge_augmentation.png "ImageAugmentation"
[image4]: ./output_images/4_class_dist_ext_training_set.png "ClassDist1"
[image5]: ./output_images/5_grayscaled.png "Grayscaled"
[image6]: ./output_images/6_normalized_grayscaled.png "NormalizedAndGrayscaled"
[image7]: ./output_images/7_customized_lenet.png "CustomizedLenet"
[image8]: ./output_images/8_own_test_images.png "SelfFoundTestImages"
[image9]: ./output_images/9_validation_accuracy_per_epoch.png "ValidationAccuracyPerEpoch"
---
### Step 0: load the data
The provided data are in pickle format. We load them using the following code:
```python
training_file = "./data/train.p"
valid_file= "./data/valid.p"
testing_file = "./data/test.p"
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(valid_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
```

#### 1) Data Set Summary & Exploration
```python
# Number of training examples
n_train = len(X_train)
# Number of validation examples
n_validation = len(X_valid)
# Number of testing examples.
n_test = len(X_test)
# What's the shape of an traffic sign image?
image_shape = X_train[0].shape
# How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))
```
With help of listed code, we can get the basic understanding about the provided datasets.
* The size of training set is: 34799
* The size of the validation set is: 4410
* The size of test set is: 12630
* The shape of a traffic sign image is: (32, 32, 3)
* The number of unique classes/labels in the data set is: 43

#### 2. Include an exploratory visualization of the dataset.
Here is the visualization of the 43 different classes, which are 43 different German traffic signs.

![43 different classes][image1]

Here is an exploratory visualization of the data set. It is a bar chart showing how the class distribution in the training set looks like:

![Class distribution][image2]

As can be seen, the training data is not equally distributed among the different classes. It means that the algorithm will be more fine-tuned for classes that have more examples. In order to generalize the network, the number of examples for each class should be roughly the same.

### Step1 Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Before start with training, two steps are introduced: 1) extend the original training set through generating fake data, 2) preprocess the extended training data set. 

#### Fake data generation
In order to prevent overfitting and make the network generalized, we generate fake data. The following transformations are applied.
* Translation
* Rotation
* Scaling

Here is an example of an original image and an augmented image:
![Image augmentation][image3]

Finally we extend the training set 5 times larger than the original size. The number of examples of the extended training set is 223371. The class distribution of the extended training set is as follows:

![Class distribution of the extended training set][image4]

As can be seen, the class distribution of the extended training set is more even distributed.

#### Preprocess data set
As a first step, I decided to convert the images to grayscale because shapes of traffic signs are important than their colors. Moreover, removing color information allows to increase the speed of training.

Here is an example of a traffic sign image before and after grayscaling.

![Grayscaled image][image5]

Next, I normalized the image data because it can prevent overfitting, which is discussed in the lecture. In our work, a min-max scaling is implemented. 

Here is an example of normalizing an image. Before normalization, the selected image is first grayscaled.

![Normalized grayscaled image][image6]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

Our final model is based on the structure of the well-known LeNet, which is designed for identifying the 10 handwritten digits. In our case, the classifier needs to identify 43 different German traffic signs, which are more complex than handwritten digits. Therefore, I add one more convolutional layer. In summary, it consists of the following layers:

| Layer         		  |     Description	        					    |
|:-----------------------:|:-----------------------------------------------:|
| Input         		  | 32x32x1 grayscaled image   					    |
| Convolution1 5x5x1      | 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					  |												    |
| Max pooling	      	  | 2x2 stride,  outputs 14x14x16 				    |
| Convolution2 5x5x16	  | 1x1 stride, same padding, outputs 14x14x32      |
| RELU			          |                                                 |
| Dropout			      | keep probability 0.7                            |
| Convolution3 5x5x32     | 1x1 stride, valid padding, outputs 10x10x64 	|
| RELU					  |												    |
| Max pooling	      	  | 2x2 stride,  outputs 5x5x64 				    |
| Fully connected1		  | 120 neurons      							    |
| RELU				      |         									    |
| Fully connected2		  | 84 neurons									    |
| RELU					  |												    |
| Fully connected3		  |	43 neurons									    |

![Customized LeNet][image7]


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an following configurations:

- optimizer: AdamOptimizer
- batch size: 128
- number of epochs: 100
- learning rate: 0.001
- keep probability of dropout in Convolution2: 0.7 (only for training)

The defined Tensorflow placeholders are:
```python
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
dropout_keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, 43)
```

The code for training pipeline is as follows:
```python
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_operation = optimizer.minimize(loss_operation)
```
The code for evaluation pipeline is listed:
```python
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```
First I set the number of epochs was 40. I found that the validation accuracy fluctuates around the value of 0.97. Then I set the number of epochs was 100. The similar phenomenon appears. Therefore, I decide that I will not increase the number of epochs anymore.

![ValidationAccuracyPerEpoch][image9]

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

1. First I used the classic LeNet as introduced in the lecture. However, its validation accuracy is about 0.89, which is smaller than the accuracy required in this project.

2. Then I analyzed the difference between identification of handwritten digits and traffic signs. In the latter case, an image of traffic sign contains more information e.g. shapes than an image of a handwritten digits. Therefore, I decided to use more filters in each convolutional layer. Moreover, I added an additional convolutional layer, in which dropout is used in order to prevent the overfitting issue.

My final model results were:
* validation set accuracy of: 0.976
* test set accuracy of: 0.959


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web. First I resized and plotted them:
![Self found traffic signs][image8]

As can be seen, the image of "No entry" is a standard traffic sign picture. It should be easily classified. Other images contains some background information e.g. house, sky, which make classification more challenging.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Priority road (12)    | Priority road   								|
| Yield (13)     		| Yield 										|
| Stop (14)				| Stop											|
| No entry (17)	      	| No entry					 				    |
| General caution (18)	| General caution      							|
| Turn right ahead (33)	| Turn right ahead      						|


The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. However, the quality of found images play an important role in the prediction accuracy.

#### 3. Describe how certain the model is when predicting on each of the six new images by looking at the softmax probabilities for each prediction. Provide the top six softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is as follows:
```python
# predictions
prediction = tf.nn.softmax(logits)
with tf.Session() as sess:
    # restore saved trained model
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph("./lenet_traffic_sign.meta")
    saver.restore(sess, saved_model_name)
    tests_images_class = sess.run(prediction, feed_dict={x: X_test_new_processed, dropout_keep_prob: 1.0})
    
# output top 6 Softmax probabilities for each image using tf.nn.top_k
with tf.Session() as sess:
    # restore saved trained model
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph("./lenet_traffic_sign.meta")
    saver.restore(sess, saved_model_name)
    predicts = sess.run(tf.nn.top_k(tests_images_class, k=6, sorted=True))
for i in range(len(predicts[0])):
    print("#[I] img{}, predicted label {}, predicted probability {}".format(i, predicts[1][i], predicts[0][i]))
```
The output of **predicts** is:

INFO:tensorflow:Restoring parameters from "./lenet_traffic_sign"

#[I] img0, predicted label [12  0  1  2  3  4], predicted probability [ 1.  0.  0.  0.  0. 0.]

#[I] img1, predicted label [13  0  1  2  3  4], predicted probability [ 1.  0.  0.  0.  0. 0.]

#[I] img2, predicted label [14  0  1  2  3  4], predicted probability [ 1.  0.  0.  0.  0. 0.]

#[I] img3, predicted label [17  0  1  2  3  4], predicted probability [ 1.  0.  0.  0.  0. 0.]

#[I] img4, predicted label [18  0  1  2  3  4], predicted probability [ 1.  0.  0.  0.  0. 0.]

#[I] img5, predicted label [33  0  1  2  3  4], predicted probability [ 1.  0.  0.  0.  0. 0.]

E.g. the image of "Stop" is predicted as label 14 with probability of 1.0.

### Discussion
* More research and practice needed in order to design a robust and well-performed convolutional neural network. Parameters tuning needs a lot of time and resource. A structured way of tuning parameters is preferred. Some research of it is planned.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?