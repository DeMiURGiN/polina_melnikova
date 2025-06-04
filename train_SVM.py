import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def get_hog_features(img):
    return hog(img, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys")

def load_dataset(dataset_path):
    X, y = [], []
    for label in os.listdir(dataset_path):
        label_dir = os.path.join(dataset_path, label)
        for file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (20, 20))
            features = get_hog_features(img)
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)

# Загрузка данных
X, y = load_dataset("dataset/")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Обучение модели
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

# Оценка
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Сохранение модели при желании
import joblib
joblib.dump(clf, "svm_text_recognizer.pkl")
