import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Conv2D, Dropout, Lambda, Cropping2D
from keras.preprocessing.image import img_to_array
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, EarlyStopping

from helper import *


def train_generator(data, batch_size=32, augments_per_sample=8):
    nb_data = len(data)
    while 1:  # loop over EPOCHS
        shuffle(data)
        for offset in range(0, nb_data, batch_size):  # loop over batches
            batch_samples = data.values[offset:offset + batch_size]

            images = []
            steering_angles = []
            for batch_sample in batch_samples:  # loop over samples in batch
                for n in range(augments_per_sample):  # augmentations per sample

                    image, steering_angle = load_and_augment_image(batch_sample)

                    images.append(img_to_array(image))
                    steering_angles.append(steering_angle)

            X_train = np.array(images)
            y_train = np.array(steering_angles)
            yield shuffle(X_train, y_train)


def validation_generator(data, batch_size=32):
    nb_data = len(data)
    while 1: # loop over EPOCHS
        shuffle(data)
        for offset in range(0, nb_data, batch_size):  # loop over batches
            batch_samples = data.values[offset:offset + batch_size]

            images = []
            steering_angles = []
            for batch_sample in batch_samples:  # loop over samples in batch
                # don't make any augmentations, just load the data
                image, steering_angle = load_image(batch_sample)

                images.append(img_to_array(image))
                steering_angles.append(steering_angle)

            X_valid = np.array(images)
            y_valid = np.array(steering_angles)
            yield shuffle(X_valid, y_valid)


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


if __name__ == "__main__":
    # Data to learn
    paths = ['../Driving_Data/',
             '../Driving_Data_second_track_dynamic/',
             '../Driving_Data_second_track_dynamic_2/']

    # Hyperparameters
    EPOCHS = 2
    BATCH_SIZE = 32
    AUGMENTS_PER_SAMPLE = 5

    # load, clean and split data
    driving_log = read_csv(paths)
    driving_log = clean_data(driving_log, upper_angle=2.0, zero_frac=0.1)
    train_samples, validation_samples = train_test_split(driving_log, test_size=0.2)

    # Instantiate generators and model
    gen_train = train_generator(train_samples, batch_size=BATCH_SIZE, augments_per_sample=AUGMENTS_PER_SAMPLE)
    gen_valid = validation_generator(validation_samples, batch_size=BATCH_SIZE)

    # Create Tensorboard object
    tbCallBack = TensorBoard(log_dir='./tensorboard', histogram_freq=1, write_graph=True, write_images=True)
    earlystop = EarlyStopping(monitor='val_loss', patience=2)

    # transfer learning: if there is already a model, load it. If not, instantiate new model.
    if os.path.exists('model.h5'):
        model = load_model('model.h5')
        print('Previous model loaded!')
    else:
        model = nvidia_model()

    # train model using generators
    history_object = model.fit_generator(generator=gen_train,
                                         samples_per_epoch=len(train_samples) * AUGMENTS_PER_SAMPLE,
                                         validation_data=gen_valid,
                                         nb_val_samples=len(validation_samples),
                                         verbose=2,
                                         nb_epoch=EPOCHS,
                                         callbacks=[tbCallBack, earlystop])
    # save model
    model.save('model.h5')

    # print the keys contained in the history object
    print(history_object.history.keys())

    # plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
