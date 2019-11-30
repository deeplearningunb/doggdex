import numpy as np

from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_size  = 224
num_classes = 120

train_dir = './input/train/'
test_dir  = './input/test/'
model_sav = './output/model.sav'
weights   = './resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

# def prepare_data(img_paths, img_width=image_size, img_height=image_size):
#     imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
#     img_array = np.array([img_to_array(img) for img in imgs])
#     output = preprocess_input(img_array)
#     return (output)

model = Sequential()
model.add(ResNet50(
        weights=weights,
        include_top=False,
        pooling='avg'
    ))
model.add(Dense(num_classes, activation='softmax'))

model.layers[0].trainable = False

model.compile(
    optimizer='sgd',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

data_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = data_generator.flow_from_directory(
    train_dir,
    target_size=(image_size, image_size),
    class_mode='categorical',
    batch_size=100
)

validation_generator = data_generator.flow_from_directory(
    test_dir,
    target_size=(image_size, image_size),
    class_mode='categorical',
    batch_size=100
)

model.fit_generator(
    train_generator,
    steps_per_epoch=103,
    validation_data=validation_generator,
    validation_steps=104
)

with open(model_sav, 'wb') as msave:
    pickle.dump(model, msave)