# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode
In this project, I use a neural network to clone car driving behavior. It is a supervised regression problem between the car steering angles and the road images in front of a car.

Those images were taken from three different camera angles (from the center, the left and the right of the car).

The network is based on The NVIDIA model, which has been proven to work in this problem domain.

As image processing is involved, the model is using convolutional layers for automated feature engineering.

My project includes the following files:
* model.py containing the script to create and train the model
* util.py contains functions for preprocessing of images like augmentation,rotation,flip , etc.
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model is inspired from the NVIDIA model provided. It is a deep convilution network which works well with supervised image classfication problems. The model is well documented by NVIDIA from getting the training data to step of avoiding overiftting.
* I used lambda layer to normalized input images to avoid saturation and make gradients works better.
* I have added additional dropout layer to avoid overfiting after the convolution layers
* I have added ELU for activation for each layer except output layer.
 
#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfittig.
The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I ran simulator for 8-10 rounds on a single lap to generate enough data for model to learn. I combined the data given from Udacity with the data I have generated to have a proper data for learning. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a convolution neural network model similar to the NVIDIA model I thought this model might be appropriate because of the layers defined in the arhcitecture are used by NVIDIA for end to end teting of self-driving cars. It is functional to clone the given steering behavior.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that I added dropout layer in the model.

Then I pumped in lots of data as well as have done augmentation to produce huge amount of image generation based on preprocessing techniques like augmentation. This helps model to learn in different road conditions and weather conditions.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track but to improve the driving behavior in these cases, I added some of the recovery steps while generating the data in the simulator. I also ran in the opposite direction to have a more generalized view of the road for the simulator.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes as described below.

In the end, the model looks like as follows:

* Image normalization
* Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
* Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
* Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
* Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
* Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
* Drop out (0.5)
* Fully connected: neurons: 100, activation: ELU
* Fully connected: neurons: 50, activation: ELU
* Fully connected: neurons: 10, activation: ELU
* Fully connected: neurons: 1 (output)

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![ScreenShot][https://github.com/nisarg09/CarND-Behavioral-Cloning-P3/blob/master/images/center.png]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover in case it drifts off the road. These images show what a recovery looks like:

![alt text][CarND-Behavioral-Cloning-P3/images/left.png]
![alt text][CarND-Behavioral-Cloning-P3/images/right.png]

To augment the data sat, I also flipped images and angles thinking that this would provide better learning for the model in different whether and road conditions. For example, here is an image that has then been flipped:

![alt text][CarND-Behavioral-Cloning-P3/images/flip.png]
![alt text][CarND-Behavioral-Cloning-P3/images/trans.png]

After the collection process, I had around 27k number of data points. I then preprocessed this data by

Image sizing
* the images are cropped so that model won't be trained with the sky and the car front parts.
* the images are resized to 66x200 as per NVIDIA model
* Images are normalized divided by 127.5 and subtracted by 1.0

Image Augmentation
* Randomly choose right,left or center images
* for the left image, steering angle is adjusted by +0.2
* for right image, steering angle is adjusted by -0.2
* Randomly flip image left/right
* Randomly translate image horizonatally with steering angle adjustment
* Randomly translate image vertically
* Randomly added shadows
* Randomly altering image birghtness


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by NVIDIA model. I used an adam optimizer so that manually training the learning rate wasn't necessary.
