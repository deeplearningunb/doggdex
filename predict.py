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
from sklearn.datasets import load_files
from keras.utils import np_utils

filename = 'training_oil_savemodel.sav'
file = open(filename, 'rb')
classifier = pickle.load(file)

# load InceptionResNet V2 model
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
model = InceptionResNetV2(include_top=False, weights='imagenet')
from keras.preprocessing import image
from tqdm import tqdm
def path_to_tensor(img_path):
    # load RGB image. Use image size (299, 299) for InceptionResNetV2 model
    img = image.load_img(img_path, target_size=(299, 299))
    # convert image to 3D tensor with shape (299, 299, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 299, 299, 3)
    return np.expand_dims(x, axis=0)
def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
from sklearn.datasets import load_files
def load_dataset(path, shuffle=True):
    data = load_files(path, shuffle=shuffle)
    files = data['filenames']
    labels = np_utils.to_categorical(data['target'], 14) # There are 120 dog breeds
    return files, labels
test_file, _ = load_dataset('test/', shuffle=False)
test_tensor = paths_to_tensor(test_file)
features_test = model.predict(preprocess_input(test_tensor))
np.save(open('features_test.npy', 'wb'), features_test)
X_test = np.load('features_test.npy')
result = classifier.predict(X_test)
result = result[0]
greater = -1
value = -1
i = 0
responses = []
for res in result:
    responses.append(res)
    if greater < res:
        greater = res
        value = i
    i += 1

import os
#print([dog for dog in os.walk(os.getcwd() + '/dataset/test_dataset/')])
dogs = [dog[0].split('/')[-1] for dog in os.walk(os.getcwd() + '/dataset/test_dataset/')]

dogs = [dog for dog in dogs if dog != '' or dog != 'test']

dogs.sort()

print(dogs)
print('Acurácia: ' + str(greater))
print('Raça: ' + dogs[value])
