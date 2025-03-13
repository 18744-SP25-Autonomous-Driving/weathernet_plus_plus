from __future__ import print_function, division
import os
from pathlib import Path
from typing import Callable, Optional, Union
from PIL import Image
from torchvision.datasets import VisionDataset
import pandas as pd


class BDD100K_plus(VisionDataset):
    """BDD100K_plus Dataset.

    TODO: remove labels in parentheses, relabel "undefined" from BDD100K

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
        self.img_dir = os.path.join(self.root, "images")
        self.labels_file = os.path.join(self.root, "labels.csv")

        # Instantiate image paths and labels as empty lists
        self.img_paths = []  # list of strings
        self.labels = (
            []
        )  # list of dicts, e.g. dict["weather"] = "clear", dict["road"] = "0"
        # TODO: maybe this is inefficient? could replace with list, where indices correspond to specific pipelines

        # Load the labels file using pandas
        self.df_labels = pd.read_csv(self.labels_file)
        # Populate the image paths list
        self.img_paths = [
            os.path.join(self.img_dir, img_name)
            for img_name in self.df_labels.iloc[:, 0]
        ]
        # Convert DataFrame rows to dictionaries for our labels
        self.labels = self.df_labels.iloc[:, 1:].to_dict("records")

    def __getitem__(self, index, open=False):
        # Load the image
        img_path = self.img_paths[index]
        image = Image.open(img_path).convert("RGB")
        if open:
            Image.open(img_path).show()  # debugging option to see images
        pipeline_labels = self.labels[index]  # dict[str, str]

        # Apply transformations if any
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            pipeline_labels = self.target_transform(pipeline_labels)

        return image, pipeline_labels

    def __len__(self):
        return len(self.img_paths)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]

        dataset_info = "\n".join(lines) + "\n"

        # Provides information on the number of labels for each pipeline
        # e.g. for weather, how many clear, partly cloudy, etc.
        labels_info = "\n* * * * * LABELS SPREAD * * * * *\n"
        for pipeline in self.labels[0].keys():
            labels_info += f"\n{self.df_labels[pipeline].value_counts().to_string()}\n"

        return dataset_info + labels_info


"""
# BDD100K Dataset (TESTING)
x = BDD100K_plus(
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
    download=False,
)

y = x.__getitem__(0)
print(y)
z = x.__getitem__(150, True)
"""
