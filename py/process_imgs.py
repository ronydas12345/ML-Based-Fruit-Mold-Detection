import keras, os, cv2, random, shutil
from keras.models import Sequential
from keras.layers import *

def return_progress(progress, total, title, title_2, round_p = 4):
    """Return the progress of a process based on iteration count, total size, titles, and percentage rounding amount."""
    return f"{title}: {title_2} | ({progress}/{total}) ({round(((progress/total) * 100), round_p)}%))"

path = "F:\\rony temp files\\science fair 2023\\raw\\"             # Path for raw data
data_path = "F:\\rony temp files\\science fair 2023\\data\\"       # Path for seperated data
refined_path = "F:\\rony temp files\\science fair 2023\\refined\\" # Path for refined data

cats, cats2 = ["Apple", "Banana", "Grape", "Guava", "Orange", "Pomegranate", "Strawberry"], [] # Types of fruits
for i in cats: cats2.extend([f"Fresh{i}", f"Rotten{i}"])
cats = sorted(cats2)


progress, total_size = 1, 3200
for filename in os.listdir(refined_path):
    file_path = os.path.join(refined_path, filename)
    if os.path.isfile(file_path) or os.path.islink(file_path):
        os.unlink(file_path)
    elif os.path.isdir(file_path):
        shutil.rmtree(file_path)
    print(return_progress(progress, total_size, "Clearing", filename))
    progress += 1
    
cats, cats2 = ["Apple", "Banana", "Grape", "Orange", "Pomegranate", "Strawberry"], [] # Types of fruits
for i in cats: cats2.extend([f"Fresh{i}", f"Rotten{i}"])
cats = sorted(cats2)

#Read & Process Images
imgs, img_paths, progress, total_size = [], [], 1, 3200
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