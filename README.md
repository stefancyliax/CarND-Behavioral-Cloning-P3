# SDCND Project 3: Behavioral Cloning

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
The goal of the this project was to try an end-to-end approach for a self-driving car using a provided simulator. End-to-end means that the models streches the complete chain between the camera images as input and the vehicle controls as output. There are no deliberate layers in between like a enviroment model. [NVIDIA did a similar approach using a real car](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).

###Approach
I started with reading articles and blog posts about the topic, especially [NVIDIAs paper on their End-To-End approch to self-driving cars](https://arxiv.org/abs/1604.07316). For the project we were provided with a simulator to a) collect training data as well as b) have the model drive in it.

To pass the project, the car had to drive safely around the first track.


####Dataset Challenges
The inspection of the first track lead to the some observations and challenges in the project:
- Simple, very similar graphics for the whole track. Will be easy to learn but prone to overfitting.
- Clockwise track with mostly slight left bends. Leads to an unbalanced, left-heavy dataset.
- The second Track in the simulator is very differnt looking, with lots of sharp bends and vertical variations. Generalizing from the first track to the second will be a challenge.

To tackle this overfitting-prone, unbalanced dataset a data augmentation was implemented.

####Data Augmentation And Cleaning
To provide the model with a Data augmentation tackles shortcommings in the dataset. This enables the extraction of more information which helps prevent overfitting and aids in generalization.

I implemented a generator losely based on Keras [ImageDataGenerator](https://keras.io/preprocessing/image/#imagedatagenerator). It randomly loads images from the three different cameras and randomly augments the imaged by flipping and in-/decreasing the brightness. I also experimented with shifting the images horizontally and vertically [as suggested in the NVIDIA paper](https://arxiv.org/abs/1604.07316), but it yielded little benefit and increased the learning time by a factor of 10.


TODO: Bild einfügen

A big percentage of the datapoints were driving straight as the histogram shows which leads to a big bias for going straight.


####Model Architecture
I adopted the architecture from the [NVIDIA paper](https://arxiv.org/abs/1604.07316). Since it's used by NVIDIA to drive a real car on real roads, it is probably oversized for this project. This is nother reason I had be careful about overfitting and introduced Dropout at the fully-connected layers.

The model was also extended with a cropping layer, to crop insignificant portions of the images. Then a normalization layer was added followed by a 3 1x1 convolutional layer to let the network choose the best color space automatically.
Since the model is used by NVIDIA to drive a real car on real roads, it is probably oversized for this project. This is another reason I had to be careful about overfitting. For this I introduced Dropout at the fully-connected layers.

TODO: Bild einfügen

###Training
I trained the model using the generator with 32 samples with 5 random augmentations each. This adds up to a batch size of 160. For the first track I trained the model on 8859 datapoints and generated 44295 images per epoch.

###Conclusion
- really showed the benefits of transfer learning



##Example videos
https://youtu.be/YkSBKT-wS68




This repository contains starting files for the Behavioral Cloning Project.

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting five files:
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/432/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :).

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).
