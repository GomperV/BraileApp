import cv2
import numpy as np
from keras.models import load_model

from BraileApp.main import classes

# Load the trained model
model = load_model("braille_model.h5")
save_path = r"C:\Users\PC\PycharmProjects\BraileApp\BraileApp\savepath\zdjecieTestowane"

# Function to preprocess image
def preprocess_image(image):
    processed_image = cv2.resize(image, (28, 28))
    save_path_with_extension = save_path + ".jpg"  # Specify the desired file extension
    cv2.imwrite(save_path_with_extension, processed_image)  # Save the processed image
    return processed_image

# Function to predict the class of a letter image
def predict_class(letter_image):
    preprocessed_image = preprocess_image(letter_image)
    input_image = np.expand_dims(preprocessed_image, axis=0)
    predictions = model.predict(input_image)
    predicted_class_index = np.argmax(predictions)
    predicted_class = predicted_class_index
    return predicted_class

# Function to split image into individual letters
def split_image(image):
    letter_width = 28  # Assuming each letter has a fixed width
    letter_images = []
    for i in range(0, image.shape[1], letter_width):
        letter_image = image[:, i:i+letter_width]
        letter_images.append(letter_image)
    return letter_images

# Function to convert Braille text image to English text
def convert_braille_to_english(image):
    letter_images = split_image(image)
    english_text = ""
    for letter_image in letter_images:
        predicted_class = predict_class(letter_image)
        english_text += classes[predicted_class]
    return english_text

# Load and convert the Braille text image to English text
braille_text_image_path = r"C:\Users\PC\PycharmProjects\BraileApp\BraileApp\example\braille_text.jpg"
braille_text_image = cv2.imread(braille_text_image_path)
english_text = convert_braille_to_english(braille_text_image)
print("English text:", english_text) 