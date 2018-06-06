from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from time import time
from keras.callbacks import TensorBoard

import os

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img = load_img('data/train/greek/greek.1.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

'''
# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
 i = 0
 for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='preview', save_prefix='greek', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely
'''
# begin building the model, 15 layers, output is 3d feature maps
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) # convert to 1d feature vector

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# configure the model for training, loss function is binary crossentropy (goof for binary classification)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# prepare for training
batch_size = 16

train_datagen = ImageDataGenerator( # augmentation for training data
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255) # augmentation for test data, only rescaling

# generator to read pictures and generate batches of augmented images
train_generator = train_datagen.flow_from_directory(
        'data/train',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

validation_generator = test_datagen.flow_from_directory(
        'data/valid',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')

# tensorboard
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

# start training
model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=34,
        validation_data=validation_generator,
        validation_steps=800 // batch_size,
        callbacks=[tensorboard])
model.save_weights('first_try.h5')  # always save your weights after training or during training