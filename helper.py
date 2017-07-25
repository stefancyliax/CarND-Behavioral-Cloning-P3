import numpy as np
import pandas as pd
from skimage import transform, util, color
from keras.preprocessing.image import load_img, img_to_array


def read_csv(paths):
    # read in csv from list of paths. Using pandas for easy handling
    driving_log = pd.DataFrame([])
    for path in paths:
        csv_data = pd.read_csv(path + "driving_log.csv", header=None,
                               names=['Center Image', 'Left Image', 'Right Image', 'Steering Angle',
                                      'Throttle', 'Break', 'Speed'])
        driving_log = pd.concat([driving_log, csv_data])
    return driving_log


def clean_data(driving_log, upper_angle=2.0, zero_frac=0.1):
    # clean the driving log data:

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
    print('Samples going straight:', b, ', Samples steering: ', d)
    print('Number of random samples going straight that are rescued (', zero_frac * 100, '% ):', c)
    print('Number after dropping large steering angles ( larger than +-', upper_angle, '):', e)
    print('Number of cleaned samples with rescued samples: ', f)

    return driving_log


def load_and_augment_image(sample):
    # Run randomize to determine what kind of augmentation is used.
    # There are 4 kinds of augmentations with 3 ways each
    # 1. Camera: left image | center image | right image
    # 2. Flip: normal | flipped
    # 3. Brightness: bright | normal | dark
    # 4. Horizontal shift: left | normal | right
    # 5. Vertical shift: up | normal | down

    rand = np.random.random(9)

    # 1. Camera: left image | center image | right image
    image, steering_angle = load_image(sample, rand[0])

    # 2. Flip: normal | flipped
    image, steering_angle = flip_image(image, steering_angle, rand[1])

    # 3. Brightness: bright | normal | dark
    image = brightness_image(image, rand[2])

    image = invert_image(image, rand[7])

    image = grayscale(image, rand[8])

    # 4. Horizontal shift: left | normal | right
    image = h_shift_image(image, rand[3])

    # 5. Vertical shift: up | normal | down
    image = v_shift_image(image, rand[4])

    image, steering_angle = rotate_image(image, steering_angle, rand[5])

    image, steering_angle = shear_image(image, steering_angle, rand[6])

    steering_angle = np.clip(steering_angle, -1, 1)
    return image, steering_angle


def load_image(sample, rand=0.5, steering_correction=0.15):
    # Load center, left or right image based on rand
    steering_angle = float(sample[3])
    if rand < 1 / 3:
        # Left image
        image_path = sample[1]
        steering_angle += steering_correction
    elif rand > 2 / 3:
        # Right image
        image_path = sample[2]
        steering_angle -= steering_correction
    else:
        # Center image
        image_path = sample[0]

    # Load image and steering angle
    image = load_img(image_path)
    image = np.array(img_to_array(image)) / 255
    return image, steering_angle


def flip_image(img, angle, rand=1.0):
    if rand < 0.5:
        img = np.fliplr(img)
        angle = angle * -1.0
    return img, angle


def brightness_image(img, rand=0.5):
    amount = (rand - 0.5) * 0.75  # * 255 * 0.75
    img = img + amount
    img = np.clip(img, 0, 1)
    return img


def v_shift_image(img, rand=0.5):
    amount = int((rand * 40) - 20)
    tform = transform.AffineTransform(rotation=0, shear=0, translation=(0, amount))
    img = transform.warp(img, tform)
    return img


def h_shift_image(img, rand=0.5):
    amount = int((rand * 100) - 50)
    tform = transform.AffineTransform(rotation=0, shear=0, translation=(amount, 0))
    img = transform.warp(img, tform)
    return img


def rotate_image(img, angle, rand=0.5):
    amount = (rand * 30) - 15
    img = transform.rotate(img, angle=amount)
    angle = angle - ((rand-0.5) * 0.3)
    return np.array(img), angle


def shear_image(img, angle, rand=0.5):
    amount = rand - 0.5
    tform = transform.AffineTransform(rotation=0, shear=amount, translation=(amount * 80, 0))
    img = transform.warp(img, tform)
    angle = angle - ((rand - 0.5) * 0.3)
    return img, angle


def invert_image(img, rand=1):
    if rand > 0.5:
        img = util.invert(img)
    return img


def grayscale(img, rand=1):
    if rand > 0.5:
        img = color.rgb2gray(img)
        img = np.stack([img, img, img], axis=2)
        assert img.shape == (160, 320, 3)
    return img
