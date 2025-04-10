from django.shortcuts import render
import joblib
import numpy as np
from PIL import Image
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
import os
from django.conf import settings
import joblib

# Create your views here.


def index(request):
    return render(request, 'index.html')


model_path = os.path.join(settings.BASE_DIR, 'ML_Models', 'svm_skin_model.pkl')
model = joblib.load(model_path)
# model = joblib.load('ML_Models/svm_skin_model.pkl')

# Preprocess image to match training format


def preprocess_image(image_path):
    img = Image.open(image_path).resize((128, 128))  # Match training size
    img_array = np.array(img).flatten() / 255.0  # Normalize
    return img_array


@csrf_exempt
def predict_skin_disease(request):
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']
        file_path = default_storage.save(
            'uploaded_images/' + uploaded_file.name, uploaded_file)

        # Preprocess and predict
        features = preprocess_image('media/' + file_path)
        prediction = model.predict([features])[0]

        return JsonResponse({'prediction': prediction})

    return JsonResponse({'error': 'No image uploaded'}, status=400)
