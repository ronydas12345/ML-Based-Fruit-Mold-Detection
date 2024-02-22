import cv2
import numpy as np
import tkinter as tk
from keras.models import load_model
from tkinter import filedialog, messagebox
from PIL import Image

def upload():
    f_types = [('All Files','*'), ('JPG Files', '*.jpg'), ('PNG Files','*.png')]
    
    try:
        cont = True
        model = load_model("model.h5")
    except:
        cont = False
        print("Error: No model found in directory")
        
    if cont:
        path = filedialog.askopenfilename(filetypes=f_types)
        cv2_img = cv2.imread(path) 

        cv2_img = cv2_img.resize((780, 540))
        cv2_img = cv2_img.convert("L")
        cv2_img = np.array(cv2_img)
        cv2_img = cv2_img.reshape((780, 540, 3))
        cv2_img = cv2_img / 255.0
        result = model.predict([cv2_img])[0]
    
        text = np.argmax(result), max(result)
        T.config(text = "Prediction: " + str(np.argmax(result)))
        print(text)
    
root = tk.Tk()
root.geometry("550x200")
text = "No Image Found, \nPlease Upload"

Title = tk.Label(root, text = "ML Based Mold Detection in Fruits", font = ("Arial", 25))
Paragraph = tk.Label(root, text = "Predicting and Preventing Mold Spoilage of Fruit Products", font = ("Arial", 12))
Space = tk.Label(root)

T = tk.Label(root, text = text)

b  = tk.Button(root, text = "Upload Image", command = lambda:upload())
b2 = tk.Button(root, text = "Close Window", command = root.destroy)

Title.pack()
Paragraph.pack()
Space.pack()
b.pack()
b2.pack()
T.pack()

root.mainloop()