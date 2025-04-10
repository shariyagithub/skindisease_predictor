#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import joblib
import cv2
from skimage.feature import hog
import numpy as np

# Load model
model = joblib.load("naive_bayes.pkl")

# HOG function
def extract_hog(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (512, 512))  # Ensure same size used during training
    features = hog(resized, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    return features

# Prediction logic
def predict_image(path):
    img = cv2.imread(path)
    features = extract_hog(img)
    prediction = model.predict(features.reshape(1, -1))
    return prediction[0]

# GUI logic
def upload_image():
    filepath = filedialog.askopenfilename()
    if filepath:
        img = Image.open(filepath)
        img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk

        # Predict and display result
        pred = predict_image(filepath)
        result_label.config(text=f"Predicted: {pred}")

# GUI setup
root = tk.Tk()
root.title("Handwriting OCR")

Button(root, text="Upload Image", command=upload_image).pack(pady=10)

panel = Label(root)
panel.pack()

result_label = Label(root, text="", font=("Helvetica", 16))
result_label.pack(pady=10)

root.mainloop()


# In[ ]:





# In[ ]:





# In[ ]:




