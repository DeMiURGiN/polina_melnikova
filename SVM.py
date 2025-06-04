import cv2
import numpy as np
from skimage.feature import hog
from sklearn import svm
import matplotlib.pyplot as plt

# Функция для предобработки и извлечения символов
def extract_characters(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    chars = []
    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 10 and w > 5:
            roi = thresh[y:y+h, x:x+w]
            resized = cv2.resize(roi, (20, 20), interpolation=cv2.INTER_AREA)
            chars.append(resized)
            boxes.append((x, y, w, h))

    # Сортировка слева направо
    chars = [char for _, char in sorted(zip(boxes, chars), key=lambda b: b[0][1]*1000 + b[0][0])]
    return chars, thresh

# Функция извлечения HOG-признаков
def get_hog_features(img):
    return hog(img, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), block_norm='L2-Hys')


def train_or_load_svm():
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    return clf

# Основной pipeline
def recognize(image_path):
    chars, thresh_img = extract_characters(image_path)
    clf = train_or_load_svm()

    for char_img in chars:
        features = get_hog_features(char_img).reshape(1, -1)
        pred = clf.predict(features)
        plt.imshow(char_img, cmap='gray')
        plt.title(f"Predicted: {pred}")
        plt.show()

# Пример запуска
recognize("/mnt/data/2023-07-06 11-54-41 - frame at 7m20s.jpg")#Путь к изображению