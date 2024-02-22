# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 20:47:33 2024

@author: dasro
"""

import keras, os, cv2
from keras.models import Sequential
from keras.layers import *
import numpy as np

image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

def return_progress(progress, total, title, title_2, round_p = 4):
    """Return the progress of a process based on iteration count, total size, titles, and percentage rounding amount."""
    return f"{title}: {title_2} | ({progress}/{total}) ({round(((progress/total) * 100), round_p)}%))"

def get_class(img_name): 
    """Return the class of an image based on its image name. To get the class from a full path, remove everything before the last hierarchy"""
    
    # Remove Image Extension
    for i in image_extensions:
        if img_name.find(i) != None:
            index = len(img_name) - len(i)
            img_name = img_name[:index]
            break
        
    # Remove Non-Alphabetical Characters
    img_name = list(img_name)
    new = []
    for i in img_name:
        if i.isalpha():
            new.append(i)
    
    return "".join(new)

def remove_after_last_char(str_, char):
    """Return everything in a string before and after the last instance of a character"""
    
    char_index, str_ = str_.rfind(char), list(str_)
    str2 = str_[:char_index]
    extra = str_[char_index:-1]
    extra.append(str_[-1])
    return ["".join(str2), "".join(extra)]

path = "F:\\rony temp files\\science fair 2023\\raw\\"             # Path for raw data
data_path = "F:\\rony temp files\\science fair 2023\\data\\"       # Path for seperated data
refined_path = "F:\\rony temp files\\science fair 2023\\refined\\" # Path for refined data

cats, cats2 = ["Apple", "Banana", "Grape", "Guava", "Orange", "Pomegranate", "Strawberry"], [] # Types of fruits
for i in cats: cats2.extend([f"Fresh{i}", f"Rotten{i}"])
cats = sorted(cats2)

cats_dict = {}
for i in range(len(cats)): cats_dict[cats[i]] = i # The key is the class and the value is the ID

train_folder = os.path.join(data_path, 'train')
val_folder = os.path.join(data_path, 'eval')
test_folder = os.path.join(data_path, 'test')

imgs, img_paths = [], []

#Read Seperated Images from all folders
for folder in [train_folder, val_folder, test_folder]:
    progress, total_size = 1, len(os.listdir(folder))
    print(folder)
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        img_paths.append(filepath)
        imgs.append(cv2.imread(filepath))
        print(return_progress(progress, total_size, "Reading", filename))
        progress += 1

x_test, x_train, y_test, y_train = [], [], [], []
for i in range(0, len(img_paths)):
    img_class = get_class(remove_after_last_char(img_paths[i], "\\")[1])
    
    if "train" in img_paths[i]:
        x_train.append(imgs[i])
        y_train.append(cats_dict[img_class])
        
    elif "test" in img_paths[i]:
        x_test.append(imgs[i])
        y_test.append(cats_dict[img_class])

x_test  = np.array(x_test)
x_train = np.array(x_train)
y_test  = np.array(y_test)
y_train = np.array(y_train)

"""
X_regression = np.expand_dims(np.arange(0, 1000, 5), axis=1)
y_regression = np.expand_dims(np.arange(100, 1100, 5), axis=1)
"""

#x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
#x_test  = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.reshape(x_train.shape[0], 780, 540, 3)
x_test  = x_test.reshape(x_test.shape[0], 780, 540, 3)

#product = 943,488,000
print(x_train.shape)
#x_train = x_train.reshape(2240, 540, 780, -1)
#x_test  = x_test.reshape(480, 540, 780, -1)

#x_train = x_train.reshape(1, 540, 780, 2240)
#x_test  = x_test.reshape(1, 540, 780, 480)

y_train = keras.utils.to_categorical(y_train, len(cats))
y_test  = keras.utils.to_categorical(y_test, len(cats))

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

" x should be image and y should be class of image "

# Model Creation
model = Sequential()
#model.add(Flatten(input_shape=(780, 540, 3)))

model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = (780, 540, 3)))
model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(cats), activation='softmax'))
model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adadelta(), metrics=['accuracy'])



batch_size, epochs = 16, 5
hist = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = (x_test, y_test))

model.save("model.h5")

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
