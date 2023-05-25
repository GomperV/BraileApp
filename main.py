import cv2
import pytesseract

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

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
    pass


for cls in classes:
    cls_folder_path = os.path.join(folder_path, cls)
    images = os.listdir(cls_folder_path)
    for image_name in images:
        image_path = os.path.join(cls_folder_path, image_name)
        image = cv2.imread(image_path)
        # Przetwórz obraz (zmniejszenie rozmiaru, normalizacja, itp.)
        processed_image = preprocess_image(image)
        data.append(processed_image)
        labels.append(cls)

# Konwertowanie danych do macierzy numpy
data = np.array(data)
labels = np.array(labels)

# Wymiary obrazów napisów Braille'a
img_height = 28
img_width = 28


# Podział danych na zbiór treningowy i walidacyjny
train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2)

# Budowanie modelu
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(classes), activation='softmax'))

# Kompilacja modelu
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Trenowanie modelu
model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=10)

# Zapisz wytrenowany model
model.save("braille_model.h5")

