import keras, os, cv2, random, shutil
from keras.models import Sequential
from keras.layers import *
from PIL import Image

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
    
path = "F:\\rony temp files\\science fair 2023\\raw\\"             # Path for raw data
data_path = "F:\\rony temp files\\science fair 2023\\data\\"       # Path for seperated data
refined_path = "F:\\rony temp files\\science fair 2023\\refined\\" # Path for refined data

cats, cats2 = ["Apple", "Banana", "Grape", "Guava", "Orange", "Pomegranate", "Strawberry"], [] # Types of fruits
for i in cats: cats2.extend([f"Fresh{i}", f"Rotten{i}"])
cats = sorted(cats2)

#Define Variables for showing progress
total_size, progress = 3200, 1

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
    """
    new_img = Image.fromarray(imgs[i])
    new_img.save(new_path, 'JPEG')
    """
    cv2.imwrite(new_path, imgs[i])
    
    print(return_progress(progress, total_size, "Writing", remove_after_last_char(new_path, "\\")[1]))
    progress += 1
    
