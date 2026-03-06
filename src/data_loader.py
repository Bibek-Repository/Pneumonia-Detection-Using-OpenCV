import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.config import *

# -------------------------
# Data Transformations
# -------------------------

train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------
# Load Datasets
# -------------------------

def get_dataloaders():
    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=train_transforms
    )

    val_dataset = datasets.ImageFolder(
        root=VAL_DIR,
        transform=val_test_transforms
    )

    test_dataset = datasets.ImageFolder(
        root=TEST_DIR,
        transform=val_test_transforms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader, test_loader