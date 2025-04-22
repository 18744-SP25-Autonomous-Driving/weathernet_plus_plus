## File Descriptions

*   `README.md`: This file.
*   `weathernet.py`: base WeatherNet model {fog, glare, weather, timeofday}
*   `weathernetplusplus.py`: WeatherNet++ model with all seven pipelines
*   `mtl_weathernet.py`: multi-task learning model
*   `weathernet_transformer.py`: transformer model
*   `train.ipynb`: Notebook for model training. Optionally saves model weights.
*   `evaluate.ipynb`: Notebook for model evaluation. Can load checkpointed model weights
*   `data/`: directory containing images, labels, and helper files
*   `data/BDD100K_plus.py` Our custom dataset. It's a subclass of VisionDataset, and loosely follows CIFAR10's format.
*   `utils/set_seed.py`: Sets the seed for reproducibility.

## Setup Instructions
Extract the [images.tar](https://drive.google.com/file/d/1qQ2wQJhw_I9per92ecVMG2CO7TBnNIyJ/view?usp=sharing) archive into the data subdirectory, so that it looks like `weathernet_plus_plus/data/images/train/0a0a0b1a-7c39d841.jpg` etc.

Extract the [checkpoints.tar](https://drive.google.com/file/d/1i7jvoEkXa9ksxtnSVqUApQXBnz9WPcmx/view?usp=sharing) archive into the base project directory, so that it looks like `weathernet_plus_plus/checkpoints/mtl_0.pth` etc.

Run `conda env create -f environment.yml` to create a conda environment with the required packages/libraries. 

Then run `conda activate tensorboard-gpu` to activate the conda environment

## Directory Structure
```
weathernet_plus_plus/
├── checkpoints/            # Model weights
├── data/                   # Images, labels, helper files
│   ├── images/
│   │   ├── test/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   └── BDD100K_plus.py     # Custom dataset
├── runs/                   # TensorBoard logging
├── utils/                  # Helper functions
├── environment.yml         # Set up conda environment
├── evaluate.ipynb          # Evaluate model
├── README.md               # This file
└── train.ipynb             # Train model
```

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