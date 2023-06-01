from flask import Flask, render_template, request
from keras.models import load_model
import cv2
import numpy as np
from test import split_image, predict_class

app = Flask(__name__)

# Load the trained model
model = load_model("braille_model.h5")
save_path = "savepath/zdjecieTestowane"

# Preprocess image and predict class
def preprocess_image(image):
    processed_image = cv2.resize(image, (28, 28))
    save_path_with_extension = save_path + ".jpg"
    cv2.imwrite(save_path_with_extension, processed_image)
    return processed_image


# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')


# Route for image upload and prediction
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    letter_images = split_image(image)  # Split the image into individual letters

    predicted_classes = []
    for letter_image in letter_images:
        predicted_class = predict_class(letter_image, model)
        predicted_classes.append(predicted_class)

    return render_template('letters.html', predicted_classes=predicted_classes)


if __name__ == '__main__':
    app.run(debug=True)
