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

####Data Aquisition
To aquire training data I used a Xbox 360 controller and drove 4 laps and tried to keep in the center of the track. I also recorded 2 laps drivnig the track in reverse direction. Finally I recorded recovery scenarios where I drove the vehicle from the left or right side back to the center of the track.

####Data Augmentation And Cleaning
To provide the model with a Data augmentation tackles shortcommings in the dataset. This enables the extraction of more information which helps prevent overfitting and aids in generalization.

I implemented a generator losely based on Keras [ImageDataGenerator](https://keras.io/preprocessing/image/#imagedatagenerator). It randomly loads images from the three different cameras and randomly augments the imaged by flipping and in-/decreasing the brightness. I also experimented with shifting the images horizontally and vertically [as suggested in the NVIDIA paper](https://arxiv.org/abs/1604.07316), but it yielded little benefit and increased the learning time by a factor of 10.


TODO: Bild einfügen

A big percentage of the datapoints were driving straight as the histogram shows which leads to a big bias for going straight. I implemented a function to clean the data by dropping most, but not all datapoint with a steering angle of 0. The function also allows to drop steering angles above a margin but this wasn't used in the end.
```python
from test import test
```


TODO: Histogramme einfügen

Finally the images are cropped and normalized in the Keras model.


####Model Architecture
The model used as derived from the architecture used by [NVIDIA paper](https://arxiv.org/abs/1604.07316). Since it's used to drive a real car on real roads, it is probably oversized for this project. This is another reason I had be careful about overfitting. To combat this, I introduced Dropout at the fully-connected layers.

The model was also extended with a cropping layer, to crop insignificant portions of the images. Then a normalization layer was added followed by a 3 1x1 convolutional layer to let the network choose the best color space automatically.

Since the model is used by NVIDIA to drive a real car on real roads, it is probably oversized for this project. This is another reason I had to be careful about overfitting. For this I introduced Dropout at the fully-connected layers.

As optimizer the [Adam Optimizer](https://keras.io/optimizers/#adam) with a learning rate of 0.0001.
TODO: Bild einfügen


###Training
I trained the model using the generator with 32 samples with 5 random augmentations each. This adds up to a batch size of 160.
After splitting the data into training (80%) and validation (20%) set, I trained the model on 7087 datapoints and generated 35435 images per epoch. After 3 epochs the car was able to drive the first track in the simulator.

Tests showed, that training for up to 50 epochs would result in a better performing car. This is probably due to the random data augmentation strategy. Each epoch is training on a new set of 35435 images.

TODO: Bild einfügen.
Note that the model was not able to drive the car on the challenge track with only data from the first track. This suggest that the model is not generalizing very well. A more sofisticated augementation should be able to drive the car on the second track even after never seeing it in training.
Note that I was able to drive the vehicle on the first track after only training it on data from the second track.

###Shortcomings
- I was not able to generalize the learning enough that the car could be teached to drive on track 2 after only beeing trained on track 1. For this more data augmentations would be necessary.
- I'd like to do some tests with a smaller model to better fit the problem of track 1.



##Second Track
After the first track didn't provide a big challenge I tried my luck on the second track. I recorded 4 laps driving "the ideal line" by cutting corners instead of keeping in the middle of the lane. I wanted to see if the model would be able to pick up on a more realistic driving behavior.
After splitting the data I trained the model on 7960 data points and generated 39800 random images per epoch. Due to the more complex track more epochs were necessary. I used the [Keras function EarlyStopping](https://faroit.github.io/keras-docs/1.2.2/callbacks/#earlystopping) that interrupts the training when the validation loss isn't decreasing anymore and started training with 50 epochs over night.
TODO: Bild einfügen.
EarlyStopping didn't interrupt the learning process because of the a unstable validation loss. At about 35 epochs overfitting started but the model was still learning.

After the training the model was almost able to drive the second track at a full 30mph. It was just cutting one corner to much. To eliminate this I recorded the corner again the finetuned the mode by loading the model and trained it for another 2 epochs on just the data of the corner. After that the car was able to drive the second track.

https://youtu.be/YkSBKT-wS68




###Conclusion
Really showed the benefits of transfer learning



##Example videos
