from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import pickle
import cv2
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect():
    if request.method == 'POST':
        MODEL_PATH = "kucing.model"
        PICKLE_PATH = "lb.pickle"

        filestr = request.files['image'].read()
        npimg = np.fromstring(filestr, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # pre-process the image for classification
        image = cv2.resize(image, (96, 96))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # load the trained CNN and the label binarizer
        print("[INFO] loading network...")
        model = load_model(MODEL_PATH)
        lb = pickle.loads(open(PICKLE_PATH, "rb").read())

        # classify the input image
        print("[INFO] classifying image...")
        proba = model.predict(image)[0]
        idx = np.argmax(proba)
        label = lb.classes_[idx]

        return jsonify(success=1, label=label, percent=(proba[idx] * 100))
    
    return jsonify(error=1, message='Unsupported HTTP method')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
