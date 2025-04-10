{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "f4745ce2-85e9-4670-aa0c-4145e4420269",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app '__main__'\n",
      " * Debug mode: on\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\n",
      " * Running on http://127.0.0.1:5000\n",
      "Press CTRL+C to quit\n",
      " * Restarting with watchdog (windowsapi)\n"
     ]
    },
    {
     "ename": "SystemExit",
     "evalue": "1",
     "output_type": "error",
     "traceback": [
      "An exception has occurred, use %tb to see the full traceback.\n",
      "\u001b[1;31mSystemExit\u001b[0m\u001b[1;31m:\u001b[0m 1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\sariy\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\IPython\\core\\interactiveshell.py:3585: UserWarning: To exit: use 'exit', 'quit', or Ctrl-D.\n",
      "  warn(\"To exit: use 'exit', 'quit', or Ctrl-D.\", stacklevel=1)\n"
     ]
    }
   ],
   "source": [
    "from flask import Flask, request, jsonify, render_template\n",
    "import pickle\n",
    "import numpy as np\n",
    "from skimage.io import imread\n",
    "from skimage.transform import resize\n",
    "import os\n",
    "\n",
    "app = Flask(__name__, static_folder=\"static\", template_folder=\"frontend\")\n",
    "\n",
    "# Load your trained SVM model\n",
    "model = pickle.load(open(\"svm.pkl\", \"rb\"))\n",
    "\n",
    "@app.route('/')\n",
    "def home():\n",
    "    return render_template('index.html')  # Loads your existing frontend\n",
    "\n",
    "@app.route('/predict', methods=['POST'])\n",
    "def predict():\n",
    "    if 'image' not in request.files:\n",
    "        return jsonify({'error': 'No image uploaded'}), 400\n",
    "\n",
    "    file = request.files['image']\n",
    "    img = imread(file)\n",
    "    img_resized = resize(img, (128, 128)).flatten().reshape(1, -1)\n",
    "\n",
    "    prediction = model.predict(img_resized)[0]\n",
    "    return jsonify({'prediction': prediction})\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    app.run(debug=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d10b6564-965e-4d93-a5d1-30a64911b3f8",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python (matplotlib_env)",
   "language": "python",
   "name": "matplotlib_env"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
