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
from .models import UploadedImage
from .ml_models import skin_model

# Create your views here.


def index(request):
    return render(request, 'index.html')



def preprocess_image(image_path):
    img = Image.open(image_path).resize((128, 128))  # Match training size
    img_array = np.array(img).flatten() / 255.0  # Normalize
    return img_array


def upload_view(request):
    if request.method == 'POST' and request.FILES.get('image'):
        # Save uploaded image
        skin_image = UploadedImage(image=request.FILES['image'])
        skin_image.save()
        
        # Predict
        img_path = os.path.join(settings.MEDIA_ROOT, skin_image.image.name)
        result = skin_model.predict(img_path)
        
        # Save result
        skin_image.prediction = result['prediction']
        skin_image.save()
        
        return render(request, 'index.html', {
            'image': skin_image,
            'result': result
        })
    return render(request, 'index.html')
