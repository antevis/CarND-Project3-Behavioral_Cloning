from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import csv
import cv2
import numpy as np
import models


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


def generator(folder, samples, batch_size=32, use_sides=False, use_flips=False, steer_adj=0.25, resize=False):
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

                    if resize:
                        image = cv2.resize(src=image, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                                               
                    image = hist_eq(image)

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
    dataset = prompt_for_input_categorical('Please choose dataset (udacity, local, combi): ',
                                           ['udacity', 'local', 'combi'])
    use_side_imgs = prompt_for_input_categorical('Use side images? (y/n): ', yn) == yn[0]

    if use_side_imgs:
        steer_adjustment = prompt_for_float('Please provide adjustment value (float 0 to 1): ')

    use_flip_imgs = prompt_for_input_categorical('Add flipped images? (y/n): ', yn) == yn[0]
    use_crop_imgs = prompt_for_input_categorical('Crop images? (y/n): ', yn) == yn[0]
    batch_size = prompt_for_int('Enter batch size (16, 32, etc.): ')
    epoch_count = prompt_for_int('Enter number of epochs (1, 2, 3, etc.): ')
    model_name = prompt_for_input_categorical('Please specify the model (v1, v2): ', ['v1', 'v2'])

    csv_folder = 'drive_data_{}/'.format(dataset)
    csv_records = csv_lines(csv_folder)
    train_samples, validation_samples = train_test_split(csv_records, test_size=0.2)

    resize = model_name == 'v2'

    if use_crop_imgs:
        crop_args = ((30, 12), (0, 0)) if resize else ((60, 24), (0, 0))
    else:
        crop_args = ((0, 0), (0, 0))

    train_generator = generator(csv_folder, train_samples,
                                batch_size=batch_size,
                                use_sides=use_side_imgs,
                                use_flips=use_flip_imgs,
                                steer_adj=steer_adjustment,
                                resize=resize)

    validation_generator = generator(csv_folder, validation_samples,
                                     batch_size=batch_size,
                                     use_sides=use_side_imgs,
                                     use_flips=use_flip_imgs,
                                     steer_adj=steer_adjustment,
                                     resize=resize)

    # To calculate appropriate steps_per_epoch parameter for model.fit_generator() method
    inflate_factor = 1
    if use_side_imgs:
        inflate_factor *= 3

    if use_flip_imgs:
        inflate_factor *= 2

    model = \
        models.pcnt(row=160, col=320, ch=3, crop=crop_args) if model_name == 'v1' else \
        models.pcnt_v2(row=80, col=160, ch=3, crop=crop_args)

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

        model.save('model_{}.h5'.format(model_name))


if __name__ == '__main__':
    main()
