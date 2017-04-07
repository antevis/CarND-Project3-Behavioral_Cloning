# **Behavioral Cloning** 

---

**Behavioral Cloning Project**
**Using Deep Learning to Clone Driving Behavior

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[fuzzy_shoulder]: ./examples/placeholder.png "Model Visualization"
[fuzzy_sh_activations]: ./examples/placeholder.png "Grayscaling"
[fuzzp_sh_colorplanes]: ./examples/placeholder_small.png "Recovery Image"
[track2_curve]: ./examples/placeholder_small.png "Recovery Image"
[track2_colorplanes]: ./examples/placeholder_small.png "Recovery Image"
[track2_activations]: ./examples/placeholder_small.png "Normal Image"
[triplet]: ./examples/placeholder_small.png "Flipped Image"
[triplet_flipped]: ./examples/placeholder_small.png "Flipped Image"
[triplet_track2]: ./examples/placeholder_small.png "Flipped Image"
[triplet_track2_flipped]: ./examples/placeholder_small.png "Flipped Image"
[rgb_hist]:
[hsv_hist]:
[hls_hist]:



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolutional neural network
* model.py containing the script to create and train the model
* video.mp4 - a video recording of the vehicle driving autonomously for about one lap around the track 1
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can drive itself autonomously around the track by executing 

python drive.py model.h5


####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.



###Model Architecture and Training Strategy

####1. A lot of various models and techniques have been tried, including LeNet, models by Comma.ai, NVIDIA. Even tried to apply transfer learning from VGG-16 trained for ImageNet, but with little success.

My model consists of a convolutional neural network with 1x1 and 3x3 filter sizes and depths between 3 and 64.

Model is defined in model.py, lines 14-52

The model includes ELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

Dropout with keep probability of 0.5 has been applied to all inner convolutional and fully-connected layers, except the first convolutional layer with 1x1 kernel. This helps to reduce overfitting. (model.py lines 27, 31, 35, 40, 43, 46).

Splitting data to training and validation parts with the ratio of 0.8/0.2 should also help to avoid overfitting(model.py line 198).
The model was tested by running it through the simulator and ensuring that the vehicle stays on track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 50).

####4. Appropriate training data

Sample training data [provided by Udacity](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) proved to be sufficient to generalize to the Track 1. It contains enough of reliable driving data along the Track 1 in both directions, and seems to have some recovery situations recorded as well.

For details about how I utilized the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

First, I am convinced that the problem can be solved with plain simple linear regression with steering angle being a function of left and right 'lanes' slopes, and I even managed to implement it in Keras and make the car sucessfully follow the road for some time. IMO the main difficulty here is that it's not so easy to obtain solid left and right 'lane lines' as it was in Project 1. Knowing that the next project will be again about lane finding, I decided to re-visit the idea later. Furthermore, the objective is to create a convolutional network and linear regression model probably wouldn't count here.

To derive a working deep convolutional network model, I've followed all the lessons accompanying the project, and tried a lot of various techniques and well-known models, including LeNet, models by Comma.ai and NVIDIA and transfer learning from VGG-16 trained on ImageNet.

None of the above mentioned models managed to generalize even to Track 1. The real game-changing approach was to add a convolutional layer with 1x1 kernel and 3 filters (feature maps) as the first layer of convolution. The idea is inspired by [inplementation of Vivek Yadav](https://github.com/vxy10/P3-BehaviorCloning).

It is always tricky trying to interpret how the deep neural network comes up with it's judgements, and I wouldn't directly assert that having the number of feature maps in the first convolutional layer to be equal to the number of color planes allows the model to choose the best color plane.

Here is an example of what excites each of three feature maps in the image representing the hardest part of the Track 1, where the shoulder is the least distinguishable from the track:

![alt text][fuzzy_shoulder]

Here are three color planes (RGB) of the cropped version of the image (more on cropping later):

![alt text][fuzzp_sh_colorplanes]

And here are the activations of three feature maps of our first convolutional layer:

![alt text][fuzzy_sh_activations]

From this example, one could tell that 'FeatureMap 2' is really excited about 'Color plane 0' (which is red), while 'FeatureMap 0' activates on darker values.

Second example, from Track 2:

![alt text][track2_curve]

It's color planes:

![alt text][track2_colorplanes]

And it's feature maps activations:

![alt text][track2_activations]

All color planes look almost identical - in fact, the original image looks almost grayscale, with a slight greenish bias at the lower-right.
'FeatureMap 0' confirms it's preference towards low values while 'FeatureMap 2' almost replicates red color plane (0).

It's hard to tell whether there is a direct separation of color planes going on in this layer, but it is in fact helps the model to generalize.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 14-52) consists of a convolutional neural network with the following layers and layer sizes:

| Layer                     |     Description                                   | 
|:-------------------------:|:-------------------------------------------------:| 
| Input                     | 160x320x3 RGB image                               |
| Normalizer                | 160x320x3 RGB image                               |
| Cropping                  | 60px from top, 24px from bottom, outputs 76x320x3	|
| Convolution 1x1           | 2x2 stride, VALID padding, outputs 38x160x3       |
| ELU (exponential LU)      |                                                   |
| Convolution 3x3           | 1x1 stride, VALID padding, outputs 36x158x16      |
| ELU (exponential LU)      |                                                   |
| Max pooling               | 2x2 stride,  outputs 18x79x16                     |
| Convolution 3x3           | 1x1 stride, VALID padding, outputs 16x77x32       |
| ELU (exponential LU)      |                                                   |
| Max pooling               | 2x2 stride,  outputs 8x38x32                      |
| Convolution 3x3           | 1x1 stride, VALID padding, outputs 6x36x64        |
| ELU (exponential LU)      |                                                   |
| Max pooling               | 2x2 stride,  outputs 3x18x64                      |
| Fully connected           | 1000                                              |
| Fully connected           | 100                                               |
| Fully connected           | 10                                                |
| Output, Fully connected	| 1                                                 |

Keras out-of-the-box model visualization doesn't seem to work due to incompatibility of pydot package with Python 3.

####3. Creation of the Training Set & Training Process

As mentioned above, turns out that the dataset of 8036 steering 'observations' provided by Udaicity contains enough data to create a generator pipeline for successful convergence to Track 1. However, in my case this assumes utilizing images from all three 'cameras' and horizontal flipping of all of those triplets. This effectively inflated available training data by a factor of six. Steering adjustment of +0.25 and -0.25 for left and right images respectively had been applied.

Here is an example of the triplet with steering angles adjusted for left and right images:

![alt text][triplet]

Ant this the same triplet, flipped horizontally:

![alt text][triplet_flipped]

I deliberately discarded the idea to manipulate the relative distribution of steering angle data, assuming that the distribution of steering angles may be not less important that the angles themselves for the network to infer the optimal driving behavior. That said, having a lot of zero angles considered to be a useful feature rather than an obstacle.

However, to generalize to Track 2, one obviously have to collect some driving data from it.
Track 2 has a clear separation of carriageway into two lanes for two-way driving throughout the whole track.

The lanes are rather narrow, leaving little breadth for recovery maneuvers. The serpentine nature of the track itself implies that almost the whole driving process is one continuous recovery maneuver. That said, I decided to just drive carefully trying to stay in the left lane as much is possible. I choose joystick to control the car and managed to achieve quite good driving, to my subjective judgement of cause, staying within the lane bounds throughout the whole track.

I recorded 3 full laps of driving Track 2 in both directions, collecting 20097 data points (image 'triplets'). Here is an example: 

![alt text][triplet_track2]

To balance the combined dataset, I drove Track 1 for about 3 more laps in both center-lane and recovery modes. That gave me 43256 data points in total.

As mentioned before, I used generator pipeline to yield batches of training samples, defined in model.py, lines 134 - 176. For each epoch, it first shuffles all samples records and then creates batches of a given size, yielding them one at a time. The batch cration process designed to allow choosing whether to include side images and horizontal flipping.

First, I trained model on dataset from Udacity to successfully generalize to Track 1. 

Training for 5 epochs produced validation loss of about 0.02. The validation loss here is a good measure of overfitting, and as it was better than training loss for each epoch, the model may be considered to successfully converge to Track 1. The final submission of this project contains this very model and accompaniying video has been recorded with it.

However, training on the full combined dataset with the aim to converge to Track 2 derived the unexpected results. The car successfully passed the Track 2, never even leaving the left lane, but failed to pass the Track 1. It just drove off the road at the point where the shoulder is least distinguishable from the track. It was fun to watch it veer off the tyre fence which has been presumably classified as a left lane boundary.

I considered that to generalize to more driving conditions the network probably needs more crispy visual data, to distinguish more details. I decided to apply histgram equalization to the input data to obtain more contrast images.
There are few different approaches to histogram equalization, and it would probably give more robust results if to apply it to value or lightness color planes after converting to HSV or HLS respectively, with the subsequent conversion back to RGB, but my objective was to minimize the computational cost and I just choose to equalize all off three RGB planes without any conversions back and forth.

Below are the axamples of histogram equalization:

All three planes within RGB color space:

![alt text][rgb_hist]

V-plane in HSV color space:

![alt text][hsv_hist]

L-plane in HLS color space:

![alt text][hls_hist]

Adding this to the model as a Lambda layer is tricky as OpenCV is rather picky to the input data. This is obviously possible but I decied to leave it for later and just do pre-processing outside the model (in the generator), though it requires to modifiy the pipeline in drive.py to match the same pre-processing at drive time.

After this, car could successfully pass both tracks 1 and 2 in autonomous mode, though on Track 1 its behavior is a bit different from when the model was trained on Track 1 data only. As expected, it sticks to the right side of the track, though its behavior in general is a bit more wobbly.

The submitted model is the one trained for Track 1 only, as it produce more smooth, natural and center-lane driving style for that Track 1. 
