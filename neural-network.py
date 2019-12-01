#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[ ]:


import pandas as pd
from os.path import join


# ### Variables

# In[ ]:


inputs = 'input'

img_size = 128
num_classes = 10

csv = join(inputs, 'labels.csv')
train_dir = join(inputs, 'train')
test_dir = join(inputs, 'test')

save_file = 'model.sav'


# ### Get top breeds

# In[ ]:


df = pd.read_csv(csv)
breeds_list = df['breed'].value_counts()[:num_classes]
breeds_list = breeds_list.index.tolist()
df = df[df['breed'].isin(breeds_list)]


# ### Imports for deep learning

# In[ ]:


import tensorflow as tf
from tensorflow.keras import layers, models
from keras.utils import to_categorical


# ### Model creation

# In[ ]:


model = models.Sequential()

# 1st convolutional layer
model.add(layers.Conv2D(
    32, (3, 3),
    activation='relu',
    input_shape=(img_size, img_size, 3),
    strides=2
))

model.add(layers.MaxPooling2D((2, 2)))

# 2nd convolutional layer
model.add(layers.Conv2D(
    32, (3, 3),
    activation='relu'
))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(num_classes, activation='softmax'))

model.summary()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# ### Data generators

# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

df.id += '.jpg'
size = df.shape[0]
train_size = int(size * 0.9)

train_idg = ImageDataGenerator(
    horizontal_flip=True,
    shear_range=0.2,
    zoom_range=0.2
)
valid_idg = ImageDataGenerator()

train_generator = train_idg.flow_from_dataframe(
    df[:train_size],
    directory=train_dir,
    x_col='id',
    y_col='breed',
    batch_size=32,
    class_mode='categorical',
    target_size=(img_size, img_size),
    shuffle=True
)

validation_generator = valid_idg.flow_from_dataframe(
    df[train_size:],
    directory=train_dir,
    x_col='id',
    y_col='breed',
    batch_size=64,
    class_mode='categorical',
    target_size=(img_size, img_size),
    shuffle=True
)


# ### Fit

# In[ ]:


model.fit_generator(
    train_generator,
    steps_per_epoch=1000,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=100
)


# ### Save model

# In[ ]:


# import pickle

# with open(save_file, 'wb') as save:
#     pickle.dump(model, save)


# ### Predict

# In[ ]:


# from keras.preprocessing import image
# from keras.preprocessing.image import load_image, img_to_array

# # test_idg = ImageDataGenerator()

# # test_generator = test_idg.flow_from_directory(
# #     test_dir,
# #     target_size=(img_size, img_size),
# #     batch_size=32,
# #     class_mode='categorical'
# # )

# # with open(save_file, 'rb') as save:
# #     loaded_model = pickle.load(save)

# # loss, metric = loaded_model.evaluate_generator(
# #     generator=test_generator,
# #     steps=300
# # )

# # print("Loss:", loss)
# # print("Acc.:", metric)

# img_name = input('Image name:')
# img = image.load_image(img_name, target_size=(img_size, img_size))
# img = image.img_to_array(img)

# result = loaded_model.predict(img)

