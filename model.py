import numpy as np
import cv2
import csv
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, Dropout, MaxPooling2D, Activation, Lambda, Cropping2D
from keras.preprocessing.image import load_img


# TODO: 1x1 convolutions to automatically get best color model
# TODO: 0. drive.py sends RGB images to the model; cv2.load_img() reads images in BGR format!!!!
# TODO: Horizontal and vertical shifts
# TODO: Data augmentation: brightness
# DONE: Remove data with steering angle =0
# TODO: Transfer learniong: Save model and load model every time
# TODO: Stacked histogram for visualisation



def read_csv():
    csv_data = []
    with open('../SDCND_output/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            csv_data.append(line)

    return csv_data


def import_pictures(csv_data, steering_correction=0.25):
    images = []
    steering_angle = []
    for line in csv_data:
        # Center image
        image_path = '../SDCND_output/IMG/' + line[0].split('\\')[-1]
        images.append(load_img(image_path))
        steering_angle.append(float(line[3]))
        # Left image
        image_path = '../SDCND_output/IMG/' + line[1].split('\\')[-1]
        images.append(load_img(image_path))
        steering_angle.append(float(line[3]) + steering_correction)
        # Right image
        image_path = '../SDCND_output/IMG/' + line[2].split('\\')[-1]
        images.append(load_img(image_path))
        steering_angle.append(float(line[3]) - steering_correction)

    X_train = np.array(images)
    y_train = np.array(steering_angle)
    print('After import shape X_train: ', X_train.shape, ',  Shape y_train: ', y_train.shape)

    return X_train, y_train


def augment_pictures(X_train, y_train):
    print('Before Shape X_train: ', X_train.shape, ',  Shape y_train: ', y_train.shape)

    X_aug, y_aug = [], []
    for X, y in zip(X_train, y_train):
        X_aug.append(X)
        y_aug.append(y)
        X_aug.append(cv2.flip(X, 1))
        y_aug.append(y * -1.0)

    X_aug = np.array(X_aug)
    y_aug = np.array(y_aug)
    print('After Shape X_train: ', X_aug.shape, ',  Shape y_train: ', y_aug.shape)

    return X_aug, y_aug


# X_train, y_train = import_pictures()



def nvidia_model(X_train, y_train, EPOCHS=5):
    model = Sequential()
    model.add(Cropping2D(cropping=((62, 25), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))

    model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Conv2D(64, 3, 3, subsample=(1, 1), activation='relu'))
    model.add(Conv2D(64, 3, 3, subsample=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=EPOCHS, verbose=1)

    model.save('model.h5')

    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


def generator(data, batch_size=32, steering_correction=0.25):
    # Note that a batch size results in three times the images for learning due to the left and right images
    # being used as well
    num_data = len(data)
    while 1:
        shuffle(data)
        for offset in range(0, num_data, batch_size):
            batch_samples = data[offset:offset + batch_size]

            images = []
            steering_angle = []
            for batch_sample in batch_samples:
                # Center image
                image_path = '../SDCND_output/IMG/' + batch_sample[0].split('\\')[-1]
                images.append(load_img(image_path))
                steering_angle.append(float(batch_sample[3]))
                # Left image
                image_path = '../SDCND_output/IMG/' + batch_sample[1].split('\\')[-1]
                images.append(load_img(image_path))
                steering_angle.append(float(batch_sample[3]) + steering_correction)
                # Right image
                image_path = '../SDCND_output/IMG/' + batch_sample[2].split('\\')[-1]
                images.append(load_img(image_path))
                steering_angle.append(float(batch_sample[3]) - steering_correction)

            X_train = np.array(images)
            y_train = np.array(steering_angle)
            yield shuffle(X_train, y_train)


csv_data = read_csv()
train_samples, validation_samples = train_test_split(csv_data, test_size=0.2)

X_train, y_train = import_pictures(csv_data)
X_train, y_train = augment_pictures(X_train, y_train)
