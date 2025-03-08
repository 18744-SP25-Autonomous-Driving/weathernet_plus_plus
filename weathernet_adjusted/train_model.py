"""
Main Training Code for models
"""

import os
import random
import time
import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.utils.data.dataloader
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np

from set_seed import set_random_seed
from weathernet_adjusted import AdjustedWeatherNet

# Argument parser
parser: argparse.ArgumentParser = argparse.ArgumentParser(
    description="18744 Autonomous Driving Project - Model Trainer"
)

# Define the mini-batch size, here the size is 128 images per batch
parser.add_argument(
    "--batch_size", type=int, default=128, help="Number of samples per mini-batch"
)

# Define the number of epochs for training
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")

# Define the random seed
parser.add_argument(
    "--seed", type=int, default=42, help="Random seed for reproducibility"
)
args = parser.parse_args()

# Set the number of epochs and batch size in locals
num_epochs: int = args.epochs
batch_size: int = args.batch_size
random_seed: int = args.seed

# Set Random Seed (reproducibility)
set_random_seed(random_seed)


# CIFAR10 Dataset (Images and Labels) (for testing)
# night-net will be used to predict airplane?
# glare-net will be used to predict automobile?
# precipitation-net will be used to predict bird / cat / deer?
# fog-net will be used to predict frog?
train_dataset: dsets.CIFAR10 = dsets.CIFAR10(
    root="data",
    train=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
            ),
        ]
    ),
    download=True,
)


test_dataset: dsets.CIFAR10 = dsets.CIFAR10(
    root="data",
    train=False,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
            ),
        ]
    ),
)

# Load the datasets into torch dataloaders
train_loader: DataLoader[Tuple[torch.Tensor, int]] = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)

test_loader: DataLoader[Tuple[torch.Tensor, int]] = DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False
)


model: AdjustedWeatherNet = AdjustedWeatherNet()
model_str: str = "adjusted_weathernet"

# Put the model on the GPU/accelerator if available
device: torch.device = torch.device("cpu")
if torch.cuda.is_available():
    print("Using Nvidia GPU")
    device = torch.device("cuda")

if torch.backends.mps.is_available():
    print("Using Apple Silicon GPU")
    device = torch.device("mps")

model = model.to(device)

# Define your loss and optimizer
print(model.parameters())
optimizer = torch.optim.Adam(model.parameters())


# Label mapping function
def map_labels(cifar_labels):
    """
    Convert CIFAR-10 labels to our custom labels for night-net, glare-net, and weather-net.
    """
    _night_labels = torch.where((cifar_labels == 0) | (cifar_labels == 1) | (cifar_labels == 2) | (cifar_labels == 3),
                               cifar_labels, torch.tensor(3))  # Map others to 3 (undefined)
    
    _glare_labels = (cifar_labels == 1).float()  # Binary: 1 for automobile, 0 for others
    
    _weather_labels = torch.full_like(cifar_labels, 4)  # Default to "undefined"
    for i, label in enumerate([2, 3, 4, 5, 6]):  # bird, cat, deer, dog, frog
        _weather_labels[cifar_labels == label] = i  # Map to 0-4
    
    return _night_labels.to(device), _glare_labels.to(device), _weather_labels.to(device)

# Training loop
for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    print(f"Epoch {epoch+1}")

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        night_labels, glare_labels, weather_labels = map_labels(labels)

        optimizer.zero_grad()

        # Forward pass
        predictions = model(images)

        # Compute loss
        loss = model.compute_loss(predictions, (night_labels, glare_labels, weather_labels))
        total_loss += loss.item()
        print(f"Batch Loss: {loss.item()}")

        # Backward pass
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {total_loss/len(train_loader):.4f}")