"""BDD100K Dataset class"""

from __future__ import print_function, division
import os

from pathlib import Path
from typing import Callable, Optional, Union
from PIL import Image
from torchvision.datasets import VisionDataset
import torchvision.transforms as T
import pandas as pd
import torch


class Bdd100kPlus(VisionDataset):
    """Bdd100kPlus Dataset.

    TODO: remove labels in parentheses, relabel "undefined" from BDD100K

    Label categories:
        fog: {0=no fog, 1=fog}
        glare: {0=no glare, 1=glare}
        road: {0=dry road, 1=wet road, 2=snowy road}
        traffic: {0=no traffic, 1=low/moderate traffic, 2=high traffic}
        weather: {0=clear, 1=partly cloudy, 2=overcast, 3=rainy, 4=snowy, 5=all else}
        scene: {0=residential, 1=highway, 2=city street, 3=all else}
        timeofday: {0=dawn/dusk, 1=daytime, 2=night, 3=all else}

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset where directory
            ``BDD100K_plus`` exists or will be saved to if download is set to True.
        set (str ['train', 'val', or 'test']): Creates dataset from one of the three sets.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    """

    def __init__(
        self,
        root: Union[str, Path],
        set: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        # Initialize the parent class
        super().__init__(root, transform=transform, target_transform=target_transform)

        if set not in ["train", "val", "test"]:
            raise ValueError(f"Invalid set '{set}'. Expected one of: 'train', 'val', 'test'.")

        self.set = set  # train, val, or test

        # Setup dataset specifics here
        self.img_dir = os.path.join(self.root, f"images/{self.set}")
        self.labels_file = os.path.join(self.root, f"labels/labels_{self.set}.csv")

        # Instantiate image paths and labels as empty lists
        self.img_paths = [] # list of strings
        # Load the labels file using pandas
        self.df_labels = pd.read_csv(self.labels_file)
        # Populate labels into a tensor, converting numeric strings to int64
        # all columns besides 0th column (filename) MUST BE INTS
        self.labels = torch.tensor(self.df_labels.iloc[:, 1:].values, dtype=torch.int64)
        # Populate the image paths list using 0th column (filename)
        self.img_paths = [
            os.path.join(self.img_dir, img_name)
            for img_name in self.df_labels.iloc[:, 0]
        ]

    def __getitem__(self, index, _open=False):
        # Load the image
        img_path = self.img_paths[index]
        image = Image.open(img_path).convert("RGB")
        # pipeline_labels = self.labels[index] # dict[str, str]
        pipeline_labels = self.labels[index, :]  # tensor of size (1x7)
        if _open:
            Image.open(img_path).show()  # debugging option to see images
            print(f"Image path: {img_path}")
            print(f"Labels: {pipeline_labels}")

        # Apply transformations if any
        if self.transform is not None:
            image = self.transform(image)

        if (
            self.target_transform is not None
        ):  # might be funky... idk if we use target_transforms.
            pipeline_labels = self.target_transform(pipeline_labels)

        return image, pipeline_labels

    def __len__(self):
        return len(self.img_paths)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        body.append(f"Images directory: {self.img_dir}")
        body.append(f"Labels file: {self.labels_file}")
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]

        dataset_info = "\n".join(lines) + "\n"

        # Provides information on the number of labels for each pipeline
        # e.g. for weather, how many clear, partly cloudy, etc.
        labels_info = "\n* * * * * LABELS SPREAD * * * * *\n"

        # Get column names from the dataframe (skipping the first column which is image name)
        label_columns = self.df_labels.columns[1:]

        # For each label category, show distribution
        for i, col_name in enumerate(label_columns):
            # Extract the corresponding column from the tensor
            # Count unique values and their frequencies
            unique_values, counts = torch.unique(self.labels[:, i], return_counts=True)

            # Format the output string
            value_counts = "\n".join(
                [
                    f"{val.item()}: {count.item()}"
                    for val, count in zip(unique_values, counts)
                ]
            )
            labels_info += f"\n{col_name}:\n{value_counts}\n"

        return dataset_info + labels_info


# BDD100K Dataset (TESTING)
# train_transforms = T.Compose([T.ToTensor(), T.Normalize(mean=(0.2843, 0.3026, 0.2996),
#     std=(0.1918, 0.1945, 0.1989),), T.Resize((224, 224)),])
# val_transforms = T.Compose([T.ToTensor(), T.Normalize(mean=(0.2843, 0.3026, 0.2996),
#     std=(0.1918, 0.1945, 0.1989),), T.Resize((224, 224)),])
# test_transforms = T.Compose([T.ToTensor(), T.Normalize(mean=(0.2843, 0.3026, 0.2996),
#     std=(0.1918, 0.1945, 0.1989),), T.Resize((224, 224)),])

# x = Bdd100kPlus(
#     root="data",
#     set="train",
#     transform=train_transforms,
# )
# print(x) # dataset stats

# # y = x.__getitem__(0)
# # print(y)
# z = x.__getitem__(150, _open=True)