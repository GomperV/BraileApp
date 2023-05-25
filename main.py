import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical

# Ścieżka do folderu zawierającego obrazy napisów Braille'a
folder_path = "C:/Users/PC/PycharmProjects/BraileApp/BraileApp/img"

# Lista klas (litery, cyfry itp.) w alfabecie Braille'a
classes = ["a", "b", "c", "d", "e", "f",
           "g", "h", "i", "j", "k", "l",
           "m", "n", "o", "p", "q", "r",
           "s", "t", "u", "v", "w", "x",
           "y", "z"]  # Dodaj wszystkie klasy Braille'a

# Przygotowanie danych treningowych
data = []
labels = []

# Przetwarzanie obrazów napisów Braille'a
def preprocess_image(image):
    # Przykładowa implementacja preprocessingu obrazu (zmniejszenie rozmiaru, normalizacja, itp.)
    processed_image = cv2.resize(image, (28, 28))
    return processed_image

# Iterate over image files in the folder
# Iterate over image files in the folder
class_files = {}
for image_name in os.listdir(folder_path):
    image_path = os.path.join(folder_path, image_name)
    image = cv2.imread(image_path)
    processed_image = preprocess_image(image)

    # Extract the class from the first letter of the file name
    image_class = image_name[0].lower()

    if image_class in classes:
        data.append(processed_image)
        labels.append([image_class])  # Wrap the image_class in a list

# # Print file names from each class
# for class_name, files in class_files.items():
#     print(f"Class: {class_name}")
#     for file_name in files:
#         print(file_name)
#     print()
# Konwertowanie danych do macierzy numpy
data = np.array(data)
labels = np.array(labels)

# Wymiary obrazów napisów Braille'a
img_height, img_width, img_channels = data.shape[1:]

# One-hot encoding of labels
label_mapping = {label: index for index, label in enumerate(classes)}
encoded_labels = np.array([label_mapping[label[0]] for label in labels])
encoded_labels = to_categorical(encoded_labels, num_classes=len(classes))

# Podział danych na zbiór treningowy i walidacyjny
train_data, val_data, train_labels, val_labels = train_test_split(data, encoded_labels, test_size=0.2)

# Budowanie modelu
model = Sequential()
model.add(Conv2D(128, (3, 3), activation='relu', input_shape=(img_height, img_width, img_channels)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(len(classes), activation='softmax'))

if os.path.exists("braille_model.h5"):
    print("Model juz istnieje")
else:
    print("Model file not found. Please train the model first.")
    # Kompilacja modelu
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Trenowanie modelu
    model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=10)

    # Zapisz wytrenowany model
    model.save("braille_model.h5")
