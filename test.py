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

# Function to predict the class of an image
def predict_class(image):
    preprocessed_image = preprocess_image(image)
    input_image = np.expand_dims(preprocessed_image, axis=0)
    predictions = model.predict(input_image)
    predicted_class_index = np.argmax(predictions)
    predicted_class = classes[predicted_class_index]
    return predicted_class

# Load and predict the class for a test image
test_image_path = r"C:\Users\PC\PycharmProjects\BraileApp\BraileApp\example\testimg.jpg"
test_image = cv2.imread(test_image_path)
predicted_class = predict_class(test_image)
print("Predicted class:", predicted_class)
