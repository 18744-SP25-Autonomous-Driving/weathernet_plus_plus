## File Descriptions

*   `README.md`: This file.
*   `weathernet.py`: The model itself.
*   `train.ipynb`: Notebook for model training and evaluation. Optionally saves model weights.
*   `BDD100K_plus.py` Our custom dataset. It's a subclass of VisionDataset, and loosely follows CIFAR10's format.
*   `set_seed.py`: Sets the seed for reproducibility.

## Instructions
Unzip [BDD100K_test_1-1000](https://drive.google.com/file/d/1HNdvbReeh9QNol0AU1wDmVQOttO4Hx-v/view?usp=drive_link)
into the images subdirectory, so that it looks like `data/images/cabc30fc-e7726578.jpg...` etc

## Pipelines and Labels
| Pipeline  | Label 0          | Label 1                  | Label 2          | Label 3         | Label 4        | Label 5  |
|-----------|------------------|--------------------------|------------------|-----------------|----------------|----------|
| fog       | no fog = 0       | fog = 1                  |                  |                 |                |          |
| glare     | no glare = 0     | glare = 1                |                  |                 |                |          |
| road      | dry road = 0     | wet road = 1             | snowy road = 2   |                 |                |          |
| traffic   | no traffic = 0   | low/moderate traffic = 1 | high traffic = 2 |                 |                |          |
| weather   | clear => 0       | partly cloudy => 1       | overcast => 2    | rainy => 3      | snowy => 4     | (all else) => 5 |
| scene     | residential => 0 | highway => 1             | city street => 2 | (all else) => 3 |                |          |
| timeofday | dawn/dusk => 0   | daytime => 1             | night => 2       | (all else) => 3 |                |          |