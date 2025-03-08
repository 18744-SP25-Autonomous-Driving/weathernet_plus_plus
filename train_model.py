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
from weathernet_base import weathernet

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


model: weathernet.WeatherNet = weathernet.WeatherNet()
model_str: str = "base_weathernet"

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


# Training loop
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []
total_training_time = 0
for epoch in range(num_epochs):
    # Training phase
    train_correct = 0
    train_total = 0
    train_loss = 0
    start = time.time()
    # Sets the model in training mode.
    model = model.train()

    batch_idx: int = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Put the images and labels on the GPU
        images = images.to(device)
        labels = labels.to(device)

        # Sets the gradients to zero
        optimizer.zero_grad()

        # The actual inference
        outputs = model(images)

        # Compute the loss between the predictions (outputs) and the ground-truth labels
        # and do backprop
        loss: weathernet.WeatherNet.WeatherNetLoss = model.loss(outputs, labels)
        model.backward()  # Backpropagation

        # Performs a single optimization step (parameter update)
        optimizer.step()
        train_loss += loss.total_loss

        # The outputs are one-hot labels, we need to find the actual predicted
        # labels which have the highest output confidence
        # _, predicted = outputs.max(1)
        # train_total += labels.size(0)
        # train_correct += predicted.eq(labels).sum().item()

        # # Print every 100 steps the following information
        # if (batch_idx + 1) % 100 == 0:
        #     print(
        #         "Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f Acc: %.2f%%"
        #         % (
        #             epoch + 1,
        #             num_epochs,
        #             batch_idx + 1,
        #             len(train_dataset) // batch_size,
        #             train_loss / (batch_idx + 1),
        #             100.0 * train_correct / train_total,
        #         )
        #     )

    end = time.time()
    per_epoch_training_time = end - start
    total_training_time += per_epoch_training_time
    train_loss_list.append(train_loss / (batch_idx + 1))
    train_acc_list.append(100.0 * train_correct / train_total)

    # Testing phase
    # test_correct = 0
    # test_total = 0
    # test_loss = 0
    # # Sets the model in evaluation mode
    # model = model.eval()
    # # Disabling gradient calculation is useful for inference.
    # # It will reduce memory consumption for computations.
    # with torch.no_grad():
    #     for batch_idx, (images, labels) in enumerate(test_loader):
    #         # Put the images and labels on the GPU
    #         images = images.to(device)
    #         labels = labels.to(device)
    #         # Perform the actual inference
    #         outputs = model(images)
    #         # Compute the loss
    #         loss = model.loss(outputs, labels)
    #         test_loss += loss.total_loss

    #         # The outputs are one-hot labels, we need to find the actual predicted
    #         # labels which have the highest output confidence
    #         _, predicted = torch.max(outputs.data, 1)
    #         test_total += labels.size(0)
    #         test_correct += predicted.eq(labels).sum().item()

    # print(
    #     "Test loss:",
    #     f"{test_loss / (batch_idx + 1):.4f}",
    #     f"Test accuracy: {100.0 * test_correct / test_total:.2f}%"
    # )

    # test_loss_list.append(test_loss / (batch_idx + 1))
    # test_acc_list.append(100.0 * test_correct / test_total)

    # Save the PyTorch model in .pt format
    os.makedirs(f"ckpt/{model_str}/", exist_ok=True)
    path = f"ckpt/{model_str}/{model_str}_{epoch}.pt"
    torch.save(model.state_dict(), path)


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


total_param_list = [p for p in model.parameters()]
trainable_param_list = [p for p in model.parameters() if p.requires_grad]
num_trainable_params = 0
for param in trainable_param_list:
    num_trainable_params += np.prod(param.cpu().data.numpy().shape)

# print(f"Overall training accuracy at the end of {num_epochs} epochs : {train_acc_list[-1]:.4f}%")
# print(f"Overall test accuracy at the end of {num_epochs} epochs : {test_acc_list[-1]:.4f}%")
print(f"Total time for training : {total_training_time:.4f} seconds")
print(f"Total number of trainable parameters : {num_trainable_params}")
