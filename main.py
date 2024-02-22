import keras
import os, cv2, random, shutil
from keras.models import Sequential
from keras.layers import *
import numpy as np

image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

def return_progress(progress, total, title, title_2, round_p = 4):
    """Return the progress of a process based on iteration count, total size, titles, and percentage rounding amount."""
    return f"{title}: {title_2} | ({progress}/{total}) ({round(((progress/total) * 100), round_p)}%))"

def get_class(img_name): 
    """Return the class of an image based on its image name"""
    
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
    

# Change Path and Directory Values Here
path = "D:\\rony temp files\\science fair 2023\\raw\\"             # Path for raw data
data_path = "D:\\rony temp files\\science fair 2023\\data\\"       # Path for seperated data
refined_path = "D:\\rony temp files\\science fair 2023\\refined\\" # Path for refined data
cats, cats2 = ["Apple", "Banana", "Orange", "Strawberry"], []      # Types of fruits
total_size = 12283                                                 # Number of images in raw dataset

"""To change this into a generic model, remove lines 51 - 52, the cats2 variable, and insert all the classes on line 46"""

for i in cats: cats2.extend([f"Fresh{i}", f"Rotten{i}"])
cats = sorted(cats2)

progress = 1
for filename in os.listdir(refined_path):
    file_path = os.path.join(refined_path, filename)
    if os.path.isfile(file_path) or os.path.islink(file_path):
        os.unlink(file_path)
    elif os.path.isdir(file_path):
        shutil.rmtree(file_path)
    print(return_progress(progress, total_size, "Clearing", filename))
    progress += 1
    
cats, cats2 = ["Apple", "Banana", "Orange", "Strawberry"], [] # Types of fruits
for i in cats: cats2.extend([f"Fresh{i}", f"Rotten{i}"])
cats = sorted(cats2)

#Read & Process Images
imgs, img_paths, progress = [], [], 1
for cat in cats:
    folder_path = path + cat + "\\"
    for img_path in os.listdir(folder_path):
        img = cv2.imread(folder_path + img_path)
        img = cv2.resize(img, (780, 540), interpolation = cv2.INTER_LINEAR) # Resize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Grayscale
        cv2.imwrite(os.path.join(refined_path, img_path), img)
        
        imgs.append(img)
        img_paths.append(os.path.join(refined_path, img_path))
        
        print(return_progress(progress, total_size, "Processing", img_path))
        progress += 1

#Define Variables for showing progress
progress = 1

imgs, img_paths = [], []
#Read Refined Images from folder
for filename in os.listdir(refined_path):
    filepath = os.path.join(refined_path, filename)
    img_paths.append(filepath)
    imgs.append(cv2.imread(filepath))
    print(return_progress(progress, total_size, "Reading", filename))
    progress += 1
    
        
#Split data
train_folder = os.path.join(data_path, 'train')
val_folder = os.path.join(data_path, 'eval')
test_folder = os.path.join(data_path, 'test')

print(imgs)

# Sets the random seed 
random.seed(42)

# Shuffle the list of image filenames

indexes = list(range(0, len(imgs)))
random.shuffle(indexes)
imgs2, paths2 = [], []
for i in indexes:
    paths2.append(img_paths[i])
    imgs2.append(imgs[i])
    
imgs, img_paths = imgs2, paths2

# Determine the number of images for each set
train_size = int(len(imgs) * 0.7)
val_size   = int(len(imgs) * 0.15)
test_size  = int(len(imgs) * 0.15)
progress = 1

# Create destination folders if they don't exist
for folder_path in [train_folder, val_folder, test_folder]: 
    if not os.path.exists(folder_path): os.makedirs(folder_path)

# Copy image files to destination folders
for i, f in enumerate(img_paths):
    if i < train_size: dest_folder = train_folder
    elif i < train_size + val_size: dest_folder = val_folder
    else: dest_folder = test_folder
    new_path = dest_folder + remove_after_last_char(f, "\\")[1]
    cv2.imwrite(new_path, imgs[i])
    
    print(return_progress(progress, total_size, "Writing", remove_after_last_char(new_path, "\\")[1]))
    progress += 1
    
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

"""When using a different dataset, change the shape data accordingly"""
x_train = x_train.reshape(x_train.shape[0], 780, 540, 3)
x_test  = x_test.reshape(x_test.shape[0], 780, 540, 3)
print(x_train.shape)

y_train = keras.utils.to_categorical(y_train, len(cats))
y_test  = keras.utils.to_categorical(y_test, len(cats))

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

""" X should be image and Y should be class of image """

"""To detect undefined variables in the IDE, add 'tk.' before the model creation
lines and change the keras import to not be a star import"""

# Model Creation
model = Sequential()

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


"""If your device is not powerful, decrease the batch size and increase the epoch size for the same results.
Note that this will take considerably longer to run."""
batch_size, epochs = 128, 75
history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = (x_test, y_test))

model.save("model.h5")

score = model.evaluate(x_test, y_test, verbose=0)

"""Tune parameters and add more metrics for higher accuracy"""
print('Test loss:', score[0])
print('Test accuracy:', score[1])

"""
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
"""