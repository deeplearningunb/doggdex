# Organizing dataset into classes (folders)

## Getting all data files and classifing it

filename = 'labels.csv'
file = open(filename)

data = {}
rows = set(file.read().split('\n'))
for row in rows:
    if row:
        splited = row.split(',')
        dogfile = splited[0]
        folder = splited[1]

        if data.get(folder):
            data[folder].append(dogfile)
        else:
            data[folder] = []
            data[folder].append(dogfile)

del data['breed']

## Creating all differents folders and separating files

import os

current_path = 'dataset/training_dataset/train/'
dataset_test_path = 'dataset/test_dataset/test/'

for foldername in data.keys():
    new_path = current_path + foldername
    test_path = dataset_test_path + foldername

    try:
        os.mkdir(new_path)
        os.mkdir(test_path)
    except FileExistsError:
        print('Folder already exists.')

    counter_file = 0
    for file in data[foldername]:
        counter_file += 1
        if counter_file <= len(data[foldername]) * 0.7:
            os.rename(current_path + file + '.jpg', new_path + '/' + file + '.jpg')
        else:
            os.rename(current_path + file + '.jpg', test_path + '/' + file + '.jpg')
