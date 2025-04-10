from flask import Flask, render_template, request
import pickle
import numpy as np
from skimage.io import imread
from skimage.transform import resize

app = Flask(__name__)
model = pickle.load(open('svm.pkl', 'rb'))

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    if 'image' in request.files:
        file = request.files['image']
        img = imread(file)
        img_resized = resize(img, (512, 512)).flatten().reshape(1, -1)
        prediction = model.predict(img_resized)
        labels = ['Basal Cell Carcinoma', 'Melanoma', 'Normal', 'Squamous Cell Carcinoma']
        result = labels[prediction[0]]
        return render_template('index.html', result=result)
    return render_template('index.html', result='No image uploaded')

if __name__ == '__main__':
    app.run(debug=True)
