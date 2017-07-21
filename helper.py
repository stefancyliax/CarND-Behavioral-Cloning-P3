import numpy as np
import pandas as pd
from skimage import transform, color
from keras.preprocessing.image import load_img


def read_csv():
    driving_log = pd.read_csv('../SDCND_output/driving_log.csv', header=None,
                              names=['Center Image', 'Left Image', 'Right Image', 'Steering Angle',
                                     'Throttle', 'Break', 'Speed'])
    return driving_log


def clean_data(driving_log, upper_angle=0.7, zero_frac=0.1):
    # clean the driving log data:
    # 1. remove most but not all samples driving straight to reduce bias of going straight
    # 2. remove samples with large steering angle

    a = driving_log.shape[0]

    # Safe small number of sample of going straight
    zero_lines = driving_log[driving_log['Steering Angle'] == 0]
    b = zero_lines.shape[0]
    zero_lines_sample = zero_lines.sample(frac=zero_frac, random_state=42)
    c = zero_lines_sample.shape[0]

    # Drop all samples of driving straight
    driving_log = driving_log[driving_log['Steering Angle'] != 0]
    d = driving_log.shape[0]

    # Drop samples with large steering angles left and right
    driving_log = driving_log[driving_log['Steering Angle'] < upper_angle]
    driving_log = driving_log[driving_log['Steering Angle'] > - upper_angle]
    e = driving_log.shape[0]

    # Add some samples of driving straight back in
    driving_log = pd.concat([driving_log, zero_lines_sample])
    f = driving_log.shape[0]

    # Print statistics
    print('Number of samples in input data:', a)
    print('Number of samples going straight:', b)
    print('Number of random samples going straight that are kept (', zero_frac * 100, '% ):', c)
    print('Number of samples without samples going straight:', d)
    print('Without samples with large steering angles ( larger than +-', upper_angle, '):', e)
    print('Number of samples with rescued samples: ', f)

    return driving_log


def load_and_augment_image(sample):
    # Run randomize to determine what kind of augmentation is used.
    # There are 4 kinds of augmentations with 3 ways each
    # 1. Camera: left image | center image | right image
    # 2. Flip: normal | flipped
    # 3. Horizontal shift: left | normal | right
    # 4. Vertical shift: up | normal | down
    # 5. Brightness: bright | normal | dark
    rand = np.random.random(5)

    # 1. Camera: left image | center image | right image
    image, steering_angle = load_image(sample, rand[0])

    # 2. Flip: normal | flipped
    image, steering_angle = flip_image(image, steering_angle, rand[1])

    # 3. Horizontal shift: left | normal | right
    #image = h_shift_image(image, rand[2])

    # 4. Vertical shift: up | normal | down
    #image = v_shift_image(image, rand[3])

    # 5. Brightness: bright | normal | dark
    #image = brightness_image(image, rand[4])

    return image, steering_angle


def load_image(sample, rand=0.5, steering_correction=0.18):
    # Load center, left or right image based on rand
    steering_angle = float(sample[3])
    if rand < 1 / 3:
        # Left image
        image_path = '../SDCND_output/IMG/' + sample[1].split('\\')[-1]
        steering_angle += steering_correction
    elif rand > 2 / 3:
        # Right image
        image_path = '../SDCND_output/IMG/' + sample[2].split('\\')[-1]
        steering_angle -= steering_correction
    else:
        # Center image
        image_path = '../SDCND_output/IMG/' + sample[0].split('\\')[-1]

    # Load image and steering angle
    image = load_img(image_path)
    return image, steering_angle


def flip_image(img, angle, rand=1):
    if rand < 1 / 2:
        img = np.fliplr(img)
        angle = angle * -1.0
    return img, angle


def brightness_image(img, rand=0.5):
    img_hsv = color.rgb2hsv(img)
    if rand < 1 / 3:
        img_hsv[:, :, 2] = img_hsv[:, :, 2] - 0.3
    elif rand > 2 / 3:
        img_hsv[:, :, 2] = img_hsv[:, :, 2] + 0.3
    img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2], 0, 1)
    img = color.hsv2rgb(img_hsv)
    return img


def v_shift_image(img, rand=0.5):
    if rand < 1 / 3:
        tform = transform.AffineTransform(rotation=0, shear=0, translation=(0, 10))
        img = transform.warp(img, tform)
    elif rand > 2 / 3:
        tform = transform.AffineTransform(rotation=0, shear=0, translation=(0, -10))
        img = transform.warp(img, tform)
    return img


def h_shift_image(img, rand=0.5):
    if rand < 1 / 3:
        tform = transform.AffineTransform(rotation=0, shear=0, translation=(10, 0))
        img = transform.warp(img, tform)
    elif rand > 2 / 3:
        tform = transform.AffineTransform(rotation=0, shear=0, translation=(-10, 0))
        img = transform.warp(img, tform)
    return img
