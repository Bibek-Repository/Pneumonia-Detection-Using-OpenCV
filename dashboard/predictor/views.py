from django.shortcuts import render

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .model_utils import predict_image

def home(request):
    context = {}

    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        image_url = fs.url(filename)

        prediction, confidence = predict_image(fs.path(filename))

        context = {
            'image_url': image_url,
            'prediction': prediction,
            'confidence': f"{confidence:.2f}"
        }

    return render(request, 'index.html', context)
