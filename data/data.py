import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from data.BDD100K_plus import Bdd100kPlus


def get_dataloaders(
    dataset_name: str = "Bdd100kPlus",
    batch_size: int = 32,
    train_transforms=None,
    test_transforms=None,
):
    # usually, we want to split data into training, validation, and test sets
    # for simplicity, we will only use training and test sets
    if train_transforms is None:
        train_transforms = T.Compose([T.ToTensor(), T.Normalize(mean=(0.2843, 0.3026, 0.2996),
            std=(0.1918, 0.1945, 0.1989),), T.Resize((224, 224)),])
    if test_transforms is None:
        test_transforms = T.Compose([T.ToTensor(), T.Normalize(mean=(0.2843, 0.3026, 0.2996),
            std=(0.1918, 0.1945, 0.1989),), T.Resize((224, 224)),])

    if dataset_name == "Bdd100kPlus":
        trainset = Bdd100kPlus(
            root="./data", train=True, transform=train_transforms
        )
        testset = Bdd100kPlus(
            root="./data", train=False, transform=test_transforms
        )

        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False) # shuffle disabled for testloader

        # dbg_trainset = Subset(trainset, range(0, 50))  # Debugging: use only first 50 samples
        # dbg_trainloader = DataLoader(dbg_trainset, batch_size=batch_size, shuffle=True)
        # dbg_testset = Subset(testset, range(0, 50))
        # dbg_testloader = DataLoader(dbg_testset, batch_size=batch_size, shuffle=False)
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

    # return dbg_trainloader, dbg_testloader
    return trainloader, testloader