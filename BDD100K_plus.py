from __future__ import print_function, division
import os

import pickle
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union
import numpy as np
from PIL import Image

from torch.utils import check_integrity, download_and_extract_archive
from torchvision.datasets import VisionDataset

import torch
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class BDD100K_plus(VisionDataset):
    """BDD100K_plus Dataset.

    TODO: remove labels in parentheses, group "partly cloudy" and "overcast" into "cloudy"
    
    Label categories:
        fog (not implemented yet): {0=no fog, 1=fog}
        glare: {0=no glare, 1=glare}
        road: {0=dry road, 1=wet road, 2=snowy road}
        traffic: {0=no traffic, 1=low/moderate traffic, 2=high traffic}
        weather: {clear, partly cloudy, overcast, rainy, snowy, (foggy)}
        scene: {residential, highway, city street, (parking lot, gas stations, tunnel)}
        timeofday: {dawn/dusk, daytime, night}

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset where directory
            ``BDD100K_plus`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set. TODO: Need to split into train and test sets
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): UNUSED!

    """
    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        # Initialize the parent class
        super().__init__(root, transform=transform, target_transform=target_transform)
        
        self.train = train  # training set or test set
        
        # Setup dataset specifics here
        img_dir = os.path.join(self.root, 'data/images')
        labels_file = os.path.join(self.root, 'data/labels.csv')
        
        # Instantiate image paths and labels as empty lists
        self.img_paths = [] # list of strings
        self.labels = [] # list of dicts, e.g. dict["weather"] = "clear", dict["road"] = "0"
        # TODO: maybe this is inefficient? could replace with list, where indices correspond to specific pipelines
        
        # Parse labels file and populate lists
        with open(self.labels_file, 'r') as f:
            for line in f:
                # follows the header format of our CSV file
                img_name, glare, road, traffic, weather, scene, timeofday = line.strip().split(',')
                
                # TODO: after adding fog labels, switch this line in for the one above
                # img_name, fog, glare, road, traffic, weather, scene, timeofday = line.strip().split(',')

                pipeline_labels = {
                    # TODO: uncomment fog after adding fog labels
                    # "fog": fog,
                    "glare": glare,
                    "road": road,
                    "traffic": traffic,
                    "weather": weather,
                    "scene": scene,
                    "timeofday": timeofday
                }

                self.img_paths.append(os.path.join(self.img_dir, img_name))
                self.labels.append(pipeline_labels)
    
    def __getitem__(self, index):
        # Load the image
        img_path = self.img_paths[index]
        image = Image.open(img_path).convert('RGB')
        pipeline_labels = self.labels[index] # dict[str, str]
        
        # Apply transformations if any
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            pipeline_labels = self.target_transform(pipeline_labels)
            
        return image, pipeline_labels
    
    def __len__(self):
        return len(self.img_paths)