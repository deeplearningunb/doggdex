#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 09:53:12 2019
@author: cfnunes
"""

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

 # Convolutional Neural Network

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(rate=0.25))


# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Adding a third convolutional layer
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(rate=0.25))


# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dense(units = 32, activation = 'relu'))

classifier.add(Dense(units = 120, activation = 'softmax'))

classifier.add(Dropout(rate=0.25))
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

train_datagen = ImageDataGenerator(rescale = 1./255,
                                shear_range = 0.2,
                                zoom_range = 0.2,
                                horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('./dataset/training_dataset/train/',
                                                target_size = (64, 64),
                                                batch_size = 64,
                                                class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('./dataset/test_dataset/test/',
                                            target_size = (64, 64),
                                            batch_size = 64,
                                            class_mode = 'categorical')

classifier.fit_generator(training_set,
                        steps_per_epoch = 20,
                        epochs = 50,
                        validation_data = test_set,
                        validation_steps = 50)

# ========= SAVE MODEL ===============
filename = 'training_oil_savemodel.sav'
file = open(filename, 'wb')
pickle.dump(classifier, file)

file.close()

#=================== PREDICTION =================

train_datagen = ImageDataGenerator(rescale = 1./255,
                                        shear_range = 0.2,
                                        zoom_range = 0.2,
                                        horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('./dataset/training_dataset/train/',
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('./dataset/test_dataset/test/',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')

file = open(filename, 'rb')
loaded_model = pickle.load(file)

loss, metric = loaded_model.evaluate_generator(generator=test_set, steps=80)
print("Acurácia:" + str(metric))

test_image = image.load_img('test_image.png', target_size=(64, 64, 3))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

result = loaded_model.predict(test_image)
result = result[0]
classes = training_set.class_indices

counter = 0
for item in classes.keys():
    classes[item] = result[counter]
    counter += 1

results = [classes[item] for item in classes.keys()]

greater = [
    'init',
    -1
]

for breeddog in training_set.class_indices.keys():
    if training_set.class_indices[breeddog] > greater[1]:
        greater[0] = breeddog
        greater[1] = training_set.class_indices[breeddog]

print(greater)
print(training_set.class_indices)
