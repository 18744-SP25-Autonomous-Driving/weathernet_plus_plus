"""
Main Training Code for models
"""

import os
import time
import argparse
from typing import Tuple

import torch
import torch.utils.data.dataloader
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from set_seed import set_random_seed
from weathernetplusplus import WeatherNetPlusPlus

from BDD100K_plus import BDD100K_plus
import pdb

# Argument parser
parser: argparse.ArgumentParser = argparse.ArgumentParser(
    description="18744 Autonomous Driving Project - Model Trainer"
)

# Define the mini-batch size, here the size is 32 images per batch
parser.add_argument(
    "--batch_size", type=int, default=32, help="Number of samples per mini-batch"
)

# Define the number of epochs for training
parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")

# Define the random seed
parser.add_argument(
    "--seed", type=int, default=42, help="Random seed for reproducibility"
)

parser.add_argument(
    "--saved-state", type=str, default=None, help="Path to saved model state"
)


args = parser.parse_args()

# Set the number of epochs and batch size in locals
num_epochs: int = args.epochs
batch_size: int = args.batch_size
random_seed: int = args.seed
saved_state: str = args.saved_state

# Set Random Seed (reproducibility)
set_random_seed(random_seed)

### BDD100K_plus DATASET (IMAGES & LABELS) ###
train_dataset = BDD100K_plus(
    root="data",
    train=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                # experimentally determined
                mean=(0.2843, 0.3026, 0.2996),
                std=(0.1918, 0.1945, 0.1989)
            ),
            transforms.Resize((224, 224))
        ]
    ),
    download=False,
)

test_dataset = BDD100K_plus(
    root="data",
    train=False,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.2843, 0.3026, 0.2996),
                std=(0.1918, 0.1945, 0.1989)
            ),
            transforms.Resize((224, 224))
        ]
    ),
    download=False,
)

'''
# CIFAR10 Dataset (Images and Labels) (for TESTING ONLY)
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
'''

# Load the datasets into torch dataloaders
train_loader: DataLoader[Tuple[torch.Tensor, int]] = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)

test_loader: DataLoader[Tuple[torch.Tensor, int]] = DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False
)


model: WeatherNetPlusPlus = WeatherNetPlusPlus()
model_str: str = "WeatherNetPlusPlus"

# Put the model on the GPU/accelerator if available
device: torch.device = torch.device("cpu")
if torch.cuda.is_available():
    print("Using Nvidia GPU")
    device = torch.device("cuda")

if torch.backends.mps.is_available():
    print("Using Apple Silicon GPU")
    device = torch.device("mps")

if saved_state is not None:
    print(f"Loading saved state from {saved_state}")
    model.load_state_dict(torch.load(saved_state, weights_only=True))

model = model.to(device)

# Define your loss and optimizer
print(model.parameters())
optimizer = torch.optim.Adam(model.parameters())


def map_labels(labels):
    """
    Separates labels for all 7 categories: fog, glare, road, traffic, weather, scene, timeofday 
    for WeatherNetPlus
    """

    # TODO: Create enums for the pipeline categories, e.g. FOG = 0
    # technically, this could be reduced. But this makes it more readable.
    _fog_labels     = labels[:, 0]
    _glare_labels   = labels[:, 1]
    _road_labels    = labels[:, 2]
    _traffic_labels = labels[:, 3]
    _weather_labels = labels[:, 4]
    _scene_labels   = labels[:, 5]
    _night_labels   = labels[:, 6]

    return (_fog_labels.to(device), _glare_labels.to(device), _road_labels.to(device), 
            _traffic_labels.to(device), _weather_labels.to(device), _scene_labels.to(device), 
            _night_labels.to(device))


# Training loop
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []
total_training_time = 0
for epoch in range(args.epochs):
    model.train()
    training_loss = 0
    training_correct = 0
    training_total = 0
    epoch_start_time = time.time()
    print(f"Epoch {epoch+1}")

    start = time.time()
    batch_idx: int = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        (
            fog_labels,
            glare_labels,
            road_labels,
            traffic_labels,
            weather_labels,
            scene_labels,
            night_labels,
        ) = map_labels(labels)

        optimizer.zero_grad()

        # Forward pass
        (fog_pred, glare_pred, road_pred, traffic_pred, weather_pred, scene_pred, night_pred) = (
            model(images)
        )

        # breakpoint()
        # Compute loss
        loss = model.compute_loss(
            (
                fog_pred, 
                glare_pred, 
                road_pred, 
                traffic_pred, 
                weather_pred, 
                scene_pred, 
                night_pred
            ),
            (
                fog_labels,
                glare_labels,
                road_labels,
                traffic_labels,
                weather_labels,
                scene_labels,
                night_labels,
            ),
        )
        training_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, road_predicted = road_pred.max(1)
        _, traffic_predicted = traffic_pred.max(1)
        _, weather_predicted = weather_pred.max(1)
        _, scene_predicted = scene_pred.max(1)
        _, night_predicted = night_pred.max(1)
        training_total += (
            fog_labels.size(0) +
            glare_labels.size(0) +
            road_labels.size(0) +
            traffic_labels.size(0) +
            weather_labels.size(0) +
            scene_labels.size(0) +
            night_labels.size(0)
        )
        training_correct += (fog_pred.squeeze() > 0.5).eq(fog_labels).sum().item()
        training_correct += (glare_pred.squeeze() > 0.5).eq(glare_labels).sum().item()

        training_correct += (road_predicted == road_labels).sum().item()
        training_correct += (traffic_predicted == traffic_labels).sum().item()
        training_correct += (weather_predicted == weather_labels).sum().item()
        training_correct += (scene_predicted == scene_labels).sum().item()
        training_correct += (night_predicted == night_labels).sum().item()

        if (batch_idx + 1) % 5 == 0:
            print(
                f"Epoch: [{epoch + 1}/{num_epochs}], "
                f"Step: [{batch_idx + 1}/{len(train_dataset) // batch_size}], "
                f"Loss: {training_loss / (batch_idx + 1):.4f} "
                f"Acc: {100.0 * training_correct / training_total:.2f}%"
            )

    per_epoch_training_time = time.time() - start
    total_training_time += per_epoch_training_time
    train_loss_list.append(training_loss / (batch_idx + 1))
    train_acc_list.append(100 * training_correct / training_total)

    # Testing Phase
    testing_correct = 0
    testing_total = 0
    test_loss = 0
    model = model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            (
                fog_labels,
                glare_labels,
                road_labels,
                traffic_labels,
                weather_labels,
                scene_labels,
                night_labels,
            ) = map_labels(labels)

            # Forward pass
            (fog_pred, glare_pred, road_pred, traffic_pred, weather_pred, scene_pred, night_pred) = (
                model(images)
            )

            # Compute loss
            loss = model.compute_loss(
                (
                    fog_pred, 
                    glare_pred, 
                    road_pred, 
                    traffic_pred, 
                    weather_pred, 
                    scene_pred, 
                    night_pred
                ),
                (
                    fog_labels,
                    glare_labels,
                    road_labels,
                    traffic_labels,
                    weather_labels,
                    scene_labels,
                    night_labels,
                ),
            )
            test_loss += loss.item()

            # Calculate accuracy
            _, road_predicted = road_pred.max(1)
            _, traffic_predicted = traffic_pred.max(1)
            _, weather_predicted = weather_pred.max(1)
            _, scene_predicted = scene_pred.max(1)
            _, night_predicted = night_pred.max(1)
            testing_total += (
                fog_labels.size(0) +
                glare_labels.size(0) +
                road_labels.size(0) +
                traffic_labels.size(0) +
                weather_labels.size(0) +
                scene_labels.size(0) +
                night_labels.size(0)
            )
            testing_correct += (fog_pred.squeeze() > 0.5).eq(fog_labels).sum().item()
            testing_correct += (glare_pred.squeeze() > 0.5).eq(glare_labels).sum().item()

            testing_correct += (road_predicted == road_labels).sum().item()
            testing_correct += (traffic_predicted == traffic_labels).sum().item()
            testing_correct += (weather_predicted == weather_labels).sum().item()
            testing_correct += (scene_predicted == scene_labels).sum().item()
            testing_correct += (night_predicted == night_labels).sum().item()
    print(
        "Test loss:",
        f"{test_loss / (batch_idx + 1):.4f}",
        f"Test accuracy: {100.0 * testing_correct / testing_total:.2f}%",
    )
    test_loss_list.append(test_loss / (batch_idx + 1))
    test_acc_list.append(100.0 * testing_correct / testing_total)

    # checkpoint the model
    os.makedirs(f"ckpt/{model_str}/", exist_ok=True)
    path = f"ckpt/{model_str}/{model_str}_{epoch}.pt"
    torch.save(model.state_dict(), path)


# dump the collected stats
with open(f"{model_str}.csv", "w+") as csv_file:
    csv_file.write("Epoch,Train acc,Train loss,Test acc,Test loss\n")
    for epoch in range(num_epochs):
        csv_file.write(
            f"{epoch},"
            f"{train_acc_list[epoch]},"
            f"{train_loss_list[epoch]},"
            f"{test_acc_list[epoch]},"
            f"{test_loss_list[epoch]}\n"
        )

print(f"Total time for training : {total_training_time:.4f} seconds")
