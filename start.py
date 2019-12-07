#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Convolutional Neural Network

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import backend as K
import numpy as np
import pickle
from keras.utils import np_utils

# Load all dog breeds saved by train.py
Y_train = np.load('utils/train_labels.npy')

# Load all train files trained by train.py
X_train = np.load('utils/bottleneck_features_train.npy')

# Initialising the CNN
classifier = Sequential()

classifier.add(Flatten(input_shape = X_train.shape[1:]))
classifier.add(Dense(units = 28, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 14, activation = 'softmax'))
classifier.summary()

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, Y_train, validation_split=0.1, epochs=50, batch_size=128, verbose=1)

# Save model to be used in predict.py
filename = 'utils/training_oil_savemodel.sav'
file = open(filename, 'wb')
pickle.dump(classifier, file)
file.close()
