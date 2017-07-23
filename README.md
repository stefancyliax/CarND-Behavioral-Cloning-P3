# SDCND Project 3: Behavioral Cloning

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Overview

The goal of the this project was to try an end-to-end approach for a self-driving car using a provided simulator. End-to-end means that the models stretches the complete chain between the camera images as input and the vehicle controls as output. There are no deliberate layers in between like a enviroment model. [NVIDIA did a similar approach using a real car](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).

## Approach
I started with reading articles and blog posts about the topic, especially [NVIDIAs paper on their End-To-End approch to self-driving cars](https://arxiv.org/abs/1604.07316) and [Sentdex Python Plays GTA V project](https://psyber.io/). For the project we were provided with a simulator to a) collect training data as well as b) have the model drive in it.

#### Project Goal
To pass the project, the car had to drive autonomously around the first track.


#### Dataset Challenges
The inspection of track 1 lead to the some observations and challenges in the project:
- Simple, very similar graphics for the whole track. Will be easy to learn but prone to overfitting.
- Clockwise track with mostly slight left bends. Leads to an unbalanced, left-heavy dataset.
- The second Track in the simulator is very differnt looking, with lots of sharp bends and vertical variations. Generalizing from the first track to the second will be a challenge.

To tackle this overfitting-prone, unbalanced dataset a data augmentation was implemented.

#### Data Aquisition
To aquire training data I used a Xbox 360 controller and drove 4 laps and tried to keep in the center of the track. I also recorded 2 laps drivnig the track in reverse direction. Finally I recorded recovery scenarios where I drove the vehicle from the left or right side back to the center of the track.

#### Data Augmentation And Cleaning
To provide the model with a Data augmentation tackles shortcommings in the dataset. This enables the extraction of more information which helps prevent overfitting and aids in generalization.

I implemented a generator losely based on Keras [ImageDataGenerator](https://keras.io/preprocessing/image/#imagedatagenerator). It randomly loads images from the three different cameras and randomly augments the imaged by flipping and in-/decreasing the brightness. I also experimented with shifting the images horizontally and vertically [as suggested in the NVIDIA paper](https://arxiv.org/abs/1604.07316), but it yielded little benefit and increased the learning time by a factor of 10.

![generator function](https://github.com/stefancyliax/CarND-Behavioral-Cloning-P3/raw/master/pic/generator.png)

A big percentage of the datapoints were driving straight as the histogram shows which leads to a big bias for going straight. I implemented a function to clean the data by dropping most, but not all datapoint with a steering angle of 0. The function also allows to drop steering angles above a margin but this wasn't used in the end.
```python
def clean_data(driving_log, upper_angle=2.0, zero_frac=0.1):
    # clean the driving log data

    # Safe small number of sample of going straight
    zero_lines = driving_log[driving_log['Steering Angle'] == 0]
    zero_lines_sample = zero_lines.sample(frac=zero_frac, random_state=42)

    # Drop all samples of driving straight
    driving_log = driving_log[driving_log['Steering Angle'] != 0]

    # Drop samples with large steering angles left and right
    driving_log = driving_log[driving_log['Steering Angle'] < upper_angle]
    driving_log = driving_log[driving_log['Steering Angle'] > - upper_angle]

    # Add some samples of driving straight back in
    driving_log = pd.concat([driving_log, zero_lines_sample])

    return driving_log
```
(Note: shortened)

![histogram prior to cleaning](https://github.com/stefancyliax/CarND-Behavioral-Cloning-P3/raw/master/pic/histogram_input.png)

Statistic of cleaning data routine:
```python
Number of samples in input data: 8859
Samples going straight: 3202 , Samples steering:  5657
Number of random samples going straight that are rescued ( 10.0 % ): 320
Number after dropping large steering angles ( larger than +- 2.0 ): 5657
Number of cleaned samples with rescued samples:  5977
```
![histogram after cleaning](https://github.com/stefancyliax/CarND-Behavioral-Cloning-P3/raw/master/pic/histogram_clean.png)

![histogram final](https://github.com/stefancyliax/CarND-Behavioral-Cloning-P3/raw/master/pic/histogram_final.png)

Finally the images are cropped and normalized in the Keras model.


#### Model Architecture
The model used as derived from the architecture used by [NVIDIA paper](https://arxiv.org/abs/1604.07316). Since it's used to drive a real car on real roads, it is probably oversized for this project. This is another reason I had be careful about overfitting. To combat this, I introduced Dropout at the fully-connected layers.

The model was also extended with a cropping layer, to crop insignificant portions of the images. Then a normalization layer was added followed by a 3 1x1 convolutional layer to let the network choose the best color space automatically.

Since the model is used by NVIDIA to drive a real car on real roads, it is probably oversized for this project. This is another reason I had to be careful about overfitting. For this I introduced Dropout at the fully-connected layers.

As optimizer the [Adam Optimizer](https://keras.io/optimizers/#adam) with a learning rate of 0.0001.

![model architecture](https://github.com/stefancyliax/CarND-Behavioral-Cloning-P3/raw/master/pic/network_arch.JPG)

```python
def nvidia_model():
    model = Sequential()
    model.add(Cropping2D(cropping=((62, 25), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))

    # 1x1 convolution layer to automatically determine best color model
    model.add(Conv2D(3, 1, 1, subsample=(1, 1), activation='relu'))

    # NVIDIA model
    model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Conv2D(64, 3, 3, subsample=(1, 1), activation='relu'))
    model.add(Conv2D(64, 3, 3, subsample=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    optimizer = Adam(lr=1e-4)
    model.compile(optimizer=optimizer, loss='mse')
    print('Using Nvidia model with dropout')
    return model
```

#### Training
I trained the model using the generator with 32 samples with 5 random augmentations each. This adds up to a batch size of 160.
After splitting the data into training (80%) and validation (20%) set, I trained the model on 7087 datapoints and generated 35435 images per epoch. After 3 epochs the car was able to drive track 1 in the simulator, although the driving performance left room for improvement.

(Youtube Link)

[![First track after 3 epochs](http://img.youtube.com/vi/jsZM1ltgY8Q/0.jpg)](https://www.youtube.com/watch?v=jsZM1ltgY8Q)

Tests showed, that training for up to 50 epochs would result in better performace. This is probably due to the random data augmentation strategy. Each epoch is training on a new, randomly generated set of 35435 images.

![learning](https://github.com/stefancyliax/CarND-Behavioral-Cloning-P3/raw/master/pic/learning.png)


Project submission: (Youtube Link)

[![First track after lots  epochs](http://img.youtube.com/vi/AczlYRb4m-o/0.jpg)](https://www.youtube.com/watch?v=AczlYRb4m-o)

#### Shortcomings
- I was not able to generalize the learning enough for the car to drive on track 2 after only beeing trained on track 1. This suggest that the model is not generalizing very well. A more sofisticated augementation may be able to drive the car on the second track even after never seeing it in training. But it may also not be possible at all.
- Note that I was able to drive the vehicle on the first track after only training it on data from the second track, but with poor performance.
- Some tests with a smaller model on the first track would be interessting.


## Challenge Track
After the track 1 didn't provide a big challenge, I tried my luck on the second track. I recorded 4 laps driving "the racing line" by cutting corners instead of keeping in the middle of the lane. I wanted to see if the model would be able to pick up on a more realistic driving behavior.
After splitting the data I trained the model on 7960 data points and generated 39800 random images per epoch. Due to the more complex track more epochs were necessary. I used the [Keras function EarlyStopping](https://faroit.github.io/keras-docs/1.2.2/callbacks/#earlystopping) that interrupts the training when the validation loss isn't decreasing anymore and started training with 50 epochs over night.

![learning2](https://github.com/stefancyliax/CarND-Behavioral-Cloning-P3/raw/master/pic/learning2.png)

EarlyStopping didn't interrupt the learning process because of the a unstable validation loss. At about 35 epochs overfitting started but the model was still learning.

After the training the model was almost able to drive track 2 at a full 30mph. It was just cutting one corner a little to much. To fix this I recorded the corner again and  finetuned the model by loading the pretrained model and training it for another 2 epochs on just the data of the corner. After that the car was able to drive the second track flawlessly.

(Youtube Link)

[![Second track](http://img.youtube.com/vi/znv4QANUcTw/0.jpg)](https://www.youtube.com/watch?v=znv4QANUcTw)

#### Both tracks
To really explore what the NVIDIA model is capable of, I trained the model on ALL my recorded data from both tracks. I used the same setup to train as above.

In the end the model was able to learn both tracks. Interesstingly the performance on track 2 was better then on track 1.

(Youtube Link)

[![both tracks](http://img.youtube.com/vi/_v2axYfc7MU/0.jpg)](https://www.youtube.com/watch?v=_v2axYfc7MU)


## Conclusion
This was a fun project! It was very interessting to see what a deep neural network is capable of. [It learned to drive the racing line around track 2 faster than I could.](https://www.youtube.com/watch?v=YkSBKT-wS68) It only took data from four laps.

The project really showed the benefits of transfer learning, as I was able to finetune the model to go around one specific corner.

I learned how to use generators to be able to train on very large datasets.
