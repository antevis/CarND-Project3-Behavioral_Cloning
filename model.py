from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import csv
import cv2
import numpy as np


# Keras model definition.
# Using Exponential Linear Units activation (elu) as it had been proven to outperform ReLUs.
def pcnt(row, col, ch, crop):
    cnn_model = Sequential()

    # Didn't notice any good in centering around zero. Thus, just '/ 255.'
    cnn_model.add(Lambda(lambda x: x / 255., input_shape=(row, col, ch), name='normalizer'))

    cnn_model.add(Cropping2D(cropping=crop,
                             input_shape=(row, col, ch), name='cropping'))

    cnn_model.add(Conv2D(3, kernel_size=(1, 1), strides=(2, 2), padding='valid', activation='elu', name='cv0'))

    cnn_model.add(Conv2D(16, kernel_size=(3, 3), padding='valid', activation='elu', name='cv1'))
    cnn_model.add(MaxPooling2D(name='maxPool_cv1'))
    cnn_model.add(Dropout(0.5, name='dropout_cv1'))

    cnn_model.add(Conv2D(32, kernel_size=(3, 3), padding='valid', activation='elu', name='cv2'))
    cnn_model.add(MaxPooling2D(name='maxPool_cv2'))
    cnn_model.add(Dropout(0.5, name='dropout_cv2'))

    cnn_model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', activation='elu', name='cv3'))
    cnn_model.add(MaxPooling2D(name='maxPool_cv3'))
    cnn_model.add(Dropout(0.5, name='dropout_cv3'))

    cnn_model.add(Flatten())

    cnn_model.add(Dense(1000, activation='elu', name='fc1'))
    cnn_model.add(Dropout(0.5, name='dropout_fc1'))

    cnn_model.add(Dense(100, activation='elu', name='fc2'))
    cnn_model.add(Dropout(0.5, name='dropout_fc2'))

    cnn_model.add(Dense(10, activation='elu', name='fc3'))
    cnn_model.add(Dropout(0.5, name='dropout_fc3'))

    cnn_model.add(Dense(1, name='output'))

    cnn_model.compile(optimizer='adam', loss='mse')

    return cnn_model


# Prompt for limited number of options
def prompt_for_input_categorical(message, options):
    response = ''

    while response not in options:
        response = input(message)

    return response


def prompt_for_float(message):
    result = None

    while result is None:
        try:
            result = float(input(message))
        except ValueError:
            pass
    return result


def prompt_for_int(message):
    result = None

    while result is None:
        try:
            result = int(input(message))
        except ValueError:
            pass
    return result


# Read records from csv file
def csv_lines(csv_folder, file_name='driving_log.csv'):
    records = []

    csv_file = '{}{}'.format(csv_folder, file_name)
    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            records.append(line)
    return records


# Return image with given angle adjustment
def image_angle(file_name, angle, adjustment=0.0, resize_ratio=1.0):
    file_name = str(file_name)
    file_name = file_name.lstrip()
    img = cv2.imread(file_name)

    if adjustment != 0.0:
        angle += adjustment

    if resize_ratio != 1.0:
        img = cv2.resize(img, dsize=(0, 0), fx=resize_ratio, fy=resize_ratio)
    
    # convert BGR to RGB
    img = img[:, :, ::-1]

    return img, angle


# Histogram equalization. Converting to HSV or HLS and equalizing just value or lightness
# planes respectively with further converting back to RGB might be more robust, but computationally more costly.
# equalizing RGB planes also works
# used for generalizing to both tracks
def hist_eq(x, ch_to_heq=None):
    result = x.copy()
    if ch_to_heq is None:
        ch_to_heq = range(result.shape[2])

    for i in ch_to_heq:
        result[:, :, i] = cv2.equalizeHist(result[:, :, i])

    return result


def flip_img_angle(image, angle):
    image = cv2.flip(image, 1)
    angle *= -1.0

    return image, angle


def generator(folder, samples, batch_size=32, use_sides=False, use_flips=False, steer_adj=0.25):
    num_samples = len(samples)

    # Setting to use 1 file with no steering adjustments (1, [0.])
    # or 3 files with the given adjustment value for each (3, [0., steer_adj, -steer_adj])
    file_range = (1, [0.]) if not use_sides else (3, [0., steer_adj, -steer_adj])
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            x = []
            y = []
            for batch_sample in batch_samples:
                steer_angle = float(batch_sample[3])

                # This loop will have 1 or 3 iterations depending on
                # whether side images being used or not.
                for i in range(file_range[0]):

                    file_name = batch_sample[i].split('/')[-1]

                    file_name = '{}IMG/{}'.format(folder, file_name)

                    adj = file_range[1][i]

                    image, angle = image_angle(file_name=file_name,
                                               angle=steer_angle,
                                               adjustment=adj)
                                               
                    # used to generalize for both tracks
                    # image = hist_eq(image)

                    x.append(image)
                    y.append(angle)

                    if use_flips:
                        flip_img, flip_ang = flip_img_angle(image, angle)
                        x.append(flip_img)
                        y.append(flip_ang)

            x = np.array(x)
            y = np.array(y)

            yield x, y


def main():

    # prompts for input
    # yes/no response set (for convenience)
    yn = ['y', 'n']
    steer_adjustment = .0
    dataset = prompt_for_input_categorical('Please choose dataset (udacity or local): ',
                                           ['udacity', 'local'])
    use_side_imgs = prompt_for_input_categorical('Use side images? (y/n): ', yn) == yn[0]

    if use_side_imgs:
        steer_adjustment = prompt_for_float('Please provide adjustment value (float 0 to 1): ')

    use_flip_imgs = prompt_for_input_categorical('Add flipped images? (y/n): ', yn) == yn[0]
    use_crop_imgs = prompt_for_input_categorical('Crop images? (y/n): ', yn) == yn[0]
    batch_size = prompt_for_int('Enter batch size (16, 32, etc.): ')
    epoch_count = prompt_for_int('Enter number of epochs (1, 2, 3, etc.): ')

    csv_folder = 'drive_data_{}/'.format(dataset)
    csv_records = csv_lines(csv_folder)
    train_samples, validation_samples = train_test_split(csv_records, test_size=0.2)

    if use_crop_imgs:
        crop_args = ((60, 24), (0, 0))
    else:
        crop_args = ((0, 0), (0, 0))

    train_generator = generator(csv_folder, train_samples,
                                batch_size=batch_size,
                                use_sides=use_side_imgs,
                                use_flips=use_flip_imgs,
                                steer_adj=steer_adjustment)

    validation_generator = generator(csv_folder, validation_samples,
                                     batch_size=batch_size,
                                     use_sides=use_side_imgs,
                                     use_flips=use_flip_imgs,
                                     steer_adj=steer_adjustment)

    # To calculate appropriate steps_per_epoch parameter for model.fit_generator() method
    inflate_factor = 1
    if use_side_imgs:
        inflate_factor *= 3

    if use_flip_imgs:
        inflate_factor *= 2

    model = pcnt(row=160, col=320, ch=3, crop=crop_args)

    steps_per_epoch = len(train_samples) * inflate_factor / batch_size
    print('steps per epoch: {}'.format(steps_per_epoch))

    val_steps = len(validation_samples) * inflate_factor / batch_size
    print('validation steps per epoch: {}'.format(val_steps))

    proceed = prompt_for_input_categorical('Proceed? (y/n): ', yn) == yn[0]

    if proceed:
        model.fit_generator(train_generator,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=validation_generator,
                            validation_steps=val_steps,
                            epochs=epoch_count)

        model.save('model.h5')


if __name__ == '__main__':
    main()
