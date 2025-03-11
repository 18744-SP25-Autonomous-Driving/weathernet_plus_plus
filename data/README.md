## Instructions
Unzip [BDD100K_test_1-1000](https://drive.google.com/file/d/1HNdvbReeh9QNol0AU1wDmVQOttO4Hx-v/view?usp=drive_link)
into the images subdirectory, so that it looks like `data/images/cabc30fc-e7726578.jpg...` etc



## File Descriptions

*   `README.md`: This file.
*   `BDD100K_plus.py` Our custom dataset. It's a subclass of VisionDataset, and follows CIFAR10's format loosely.


## Pipelines and Labels
| Pipeline  | Label 0        | Label 1                  | Label 2          | Label 3       | Label 4        | Label 5  |
|-----------|----------------|--------------------------|------------------|---------------|----------------|----------|
| fog       | no fog = 0     | fog = 1                  |                  |               |                |          |
| glare     | no glare = 0   | glare = 1                |                  |               |                |          |
| road      | dry road = 0   | wet road = 1             | snowy road = 2   |               |                |          |
| traffic   | no traffic = 0 | low/moderate traffic = 1 | high traffic = 2 |               |                |          |
| weather   | clear          | partly cloudy            | overcast         | rainy         | snowy          | (foggy)  |
| scene     | residential    | highway                  | city street      | (parking lot) | (gas stations) | (tunnel) |
| timeofday | dawn/dusk      | daytime                  | night            |               |                |          |

Labels in (parentheses) are to be removed. The "undefined" label appears sometimes in the weather and scene labels, as provided by BDD100K.