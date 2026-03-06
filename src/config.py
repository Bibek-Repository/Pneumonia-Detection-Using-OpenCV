import torch

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

# Image parameters
IMAGE_SIZE = 224

# Paths
TRAIN_DIR = "data/raw/chest_xray/train"
VAL_DIR = "data/raw/chest_xray/val"
TEST_DIR = "data/raw/chest_xray/test"

# Model save path
import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 224

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")