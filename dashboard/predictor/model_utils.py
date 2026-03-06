import torch
from torchvision import transforms
from PIL import Image
import os
import sys

# Ensure project root is on sys.path so `src` package imports work reliably
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if proj_root not in sys.path:
    sys.path.append(proj_root)

from src.model import PneumoniaCNN
from src.config import DEVICE, IMAGE_SIZE, MODEL_PATH

# Load model once
model = PneumoniaCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    class_names = ["NORMAL", "PNEUMONIA"]

    return class_names[predicted.item()], confidence.item() * 100