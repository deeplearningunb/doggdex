from keras.preprocessing import image
from keras import backend as K
import numpy as np
import pickle
from sklearn.datasets import load_files
from keras.utils import np_utils

# Open trained data model to predict Dog image
filename = 'utils/training_oil_savemodel.sav'
file = open(filename, 'rb')
classifier = pickle.load(file)

# Load InceptionResNet V2 model
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
model = InceptionResNetV2(include_top=False, weights='imagenet')

# Save test data using InceptionResNetV2 model
# Reference: https://github.com/liyenhsu/Dog-Breed-Identification
from keras.preprocessing import image
from tqdm import tqdm
from sklearn.datasets import load_files

def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def load_dataset(path, shuffle=True):
    data = load_files(path, shuffle=shuffle)
    files = data['filenames']
    labels = np_utils.to_categorical(data['target'], 14)
    return files, labels

test_file, _ = load_dataset('assets/', shuffle=False)

test_tensor = paths_to_tensor(test_file)

# Creates predicted data and saves into file
features_test = model.predict(preprocess_input(test_tensor))
np.save(open('utils/features_test.npy', 'wb'), features_test)

# Predict data from model loaded and get index of dog breed
X_test = np.load('utils/features_test.npy')
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

# Rescue dog breed listing all valid dog breeds of this repository
import os

dogs = [dog[0].split('/')[-1] for dog in os.walk(os.getcwd() + '/dataset/test_dataset/')]
dogs = [dog for dog in dogs if dog != '' and dog != 'test']
dogs.sort()

print('Acurácia: ' + str(greater))
print('Raça: ' + dogs[value])
