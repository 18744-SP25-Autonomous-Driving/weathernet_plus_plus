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
        subset (int, optional): If provided, uses only the specified subset (0-6) from 
            the partitioned data. When None, uses all subsets.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    """

    def __init__(
        self,
        root: Union[str, Path],
        set: str = "train",
        subset: Optional[int] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        # Initialize the parent class
        super().__init__(root, transform=transform, target_transform=target_transform)

        if set not in ["train", "val", "test"]:
            raise ValueError(f"Invalid set '{set}'. Expected one of: 'train', 'val', 'test'.")
        
        if subset is not None and (subset < 0 or subset > 6):
            raise ValueError(f"Invalid subset '{subset}'. Expected None or integer between 0 and 6.")

        self.set = set  # train, val, or test
        self.subset = subset  # which subset to use (0-6), or None for all

        # Setup dataset specifics here
        self.img_dir = os.path.join(self.root, f"images/{self.set}")
        self.labels_file = os.path.join(self.root, f"labels/labels_{self.set}.csv")

        # Load the labels file using pandas
        self.df_labels = pd.read_csv(self.labels_file)
        
        # Filter by subset if specified
        if subset is not None:
            # Calculate which rows belong to this subset (each subset has 10,000 images)
            start_idx = subset * 10000
            end_idx = min(start_idx + 10000, len(self.df_labels))
            self.df_labels = self.df_labels.iloc[start_idx:end_idx]
        
        # Populate labels into a tensor, converting numeric strings to int64
        # all columns besides 0th column (filename) MUST BE INTS
        self.labels = torch.tensor(self.df_labels.iloc[:, 1:].values, dtype=torch.int64)
        
        # Populate the image paths list using 0th column (filename)
        self.img_paths = []
        for idx, img_name in enumerate(self.df_labels.iloc[:, 0]):
            if self.set == "train":
                # Determine which subset this image belongs to
                if self.subset is None:
                    # When using all subsets, calculate which subset each image belongs to
                    img_subset = min(idx // 10000, 6)
                    img_path = os.path.join(self.img_dir, f"subset_{img_subset}", img_name)
                else:
                    # When using a specific subset, all images are in that subset's directory
                    img_path = os.path.join(self.img_dir, f"subset_{self.subset}", img_name)
            else:
                # For validation and test sets, all images are in the same directory
                img_path = os.path.join(self.img_dir, img_name)

            # Check if the image file exists
            # Handles AFS's inability to untar all of the training data set.
            if not os.path.isfile(img_path):
                continue
            self.img_paths.append(img_path)

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

        if self.target_transform is not None: # might be funky... idk if we use target_transforms.
            pipeline_labels = self.target_transform(pipeline_labels)

        return image, pipeline_labels

    def __len__(self):
        return len(self.img_paths)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        body.append(f"Images directory: {self.img_dir}")
        if self.subset is not None:
            body.append(f"Using subset: {self.subset}")
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
# transforms = T.Compose([T.ToTensor(), T.Normalize(mean=(0.2843, 0.3026, 0.2996),
#     std=(0.1918, 0.1945, 0.1989),), T.Resize((224, 224)),])

# x = Bdd100kPlus(
#     root="data",
#     set="train",
#     transform=transforms,
# )
# print(x) # dataset stats

# y = x.__getitem__(0)
# print(y)
# z = x.__getitem__(1150, _open=True)
# z = x.__getitem__(11150, _open=True)
# z = x.__getitem__(57250, _open=True)