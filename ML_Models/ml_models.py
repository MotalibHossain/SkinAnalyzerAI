# detector/ml_model.py
import os
import cv2
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from django.conf import settings


class SkinDiseaseModel:
    def __init__(self):
        self.categories = ['Actinic keratoses',
                           'Basal cell carcinoma',
                           'Benign keratosis-like lesions',
                           'Chickenpox',
                           'Cowpox',
                           'Dermatofibroma',
                           'Healthy',
                           'HFMD',
                           'Measles',
                           'Melanocytic nevi',
                           'Melanoma',
                           'Monkeypox',
                           'Squamous cell carcinoma',
                           'Vascular lesions']
        self.train_path = os.path.join(settings.DATASET_PATH, 'train')
        self.model_path = os.path.join(settings.BASE_DIR, 'ML_Models', 'skin_model.joblib')
        self.model = self.load_or_train_model()

    def load_or_train_model(self):
        if os.path.exists(self.model_path):
            return joblib.load(self.model_path)
        return self.train_model()

    def extract_features(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (100, 100))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [
                            8, 8, 8], [0, 256, 0, 256, 0, 256])
        return cv2.normalize(hist, hist).flatten()

    def train_model(self):
        images, labels = [], []
        for category in self.categories:
            category_path = os.path.join(self.train_path, category)
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                images.append(img_path)
                labels.append(category)

        X = np.array([self.extract_features(img) for img in images])
        y = np.array(labels)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)

        model = SVC(kernel='rbf', probability=True)
        model.fit(X_train, y_train)

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(model, self.model_path)
        return model

    def predict(self, image_path):
        features = self.extract_features(image_path)
        proba = self.model.predict_proba([features])[0]
        return {
            'prediction': self.model.predict([features])[0],
            'confidence': max(proba)
        }


# Global instance
skin_model = SkinDiseaseModel()
