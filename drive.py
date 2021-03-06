import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model
import h5py
from keras import __version__ as keras_version
import cv2


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

cnn_version = 'v1'


def hist_eq(x, ch_to_heq=None):

    result = x.copy()
    if ch_to_heq is None:
        ch_to_heq = range(result.shape[2])

    for i in ch_to_heq:
        result[:, :, i] = cv2.equalizeHist(result[:, :, i])


    return result


# Prompt for limited number of options
def prompt_for_input_categorical(message, options):
    response = ''

    while response not in options:
        response = input(message)

    return response


def change_colorspace(x, new_color_space, ch_to_heq=None):

    if ch_to_heq is None:
        return cv2.cvtColor(x, new_color_space)
    else:
        x = cv2.cvtColor(x, new_color_space)

        x = hist_eq(x, ch_to_heq)

        return x


def yuv(x, ch_to_hec=None):

    return change_colorspace(x, cv2.COLOR_RGB2YUV, ch_to_hec)


def crop(image, crop_args=(30, 24, 0, 0)):

    result = image.copy()

    h = result.shape[0]
    w = result.shape[1]

    result = result[crop_args[0]:h - crop_args[1], crop_args[2]:w - crop_args[3], :]

    return result


class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.1, 0.002)
set_speed = 12
controller.set_desired(set_speed)


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)

        if cnn_version != 'v1':
            image_array = cv2.resize(src=image_array, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        # used to drive with the model generalized for both tracks.
        image_array = hist_eq(image_array)

        steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))

        throttle = controller.update(float(speed))

        print('{:.4f} {:.4f}'.format(steering_angle, throttle))
        send_control(steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )

    cnn_version = prompt_for_input_categorical('Please specify version: (v1,v2): ', ['v1', 'v2'])

    args = parser.parse_args()

    # check that model Keras version is same as local Keras version
    f = h5py.File(args.model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    model = load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
