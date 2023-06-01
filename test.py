import cv2
import numpy as np
from keras.models import load_model

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
def predict_class(letter_image, model):
    preprocessed_image = preprocess_image(letter_image)
    input_image = np.expand_dims(preprocessed_image, axis=0)
    predictions = model.predict(input_image)
    predicted_class_index = np.argmax(predictions)
    predicted_class = chr(predicted_class_index + 97)  # Convert class index to ASCII character (a = 0, b = 1, c = 2, ...)
    return predicted_class

# Function to split image into individual letters
def split_image(image):
    letter_width = 28  # Assuming each letter has a fixed width
    letter_images = []
    for i in range(0, image.shape[1], letter_width):
        letter_image = image[:, i:i+letter_width]
        letter_images.append(letter_image)
    return letter_images
