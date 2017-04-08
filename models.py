from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D


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


# 1px stride in first 1x1 convolutional layer
def pcnt_v2(row, col, ch, crop):
    cnn_model = Sequential()

    # Didn't notice any good in centering around zero. Thus, just '/ 255.'
    cnn_model.add(Lambda(lambda x: x / 255., input_shape=(row, col, ch), name='normalizer'))

    cnn_model.add(Cropping2D(cropping=crop,
                             input_shape=(row, col, ch), name='cropping'))

    cnn_model.add(Conv2D(3, kernel_size=(1, 1), padding='valid', activation='elu', name='cv0'))

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

