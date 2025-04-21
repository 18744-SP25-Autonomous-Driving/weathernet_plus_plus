"""
This model is a multi-task learning (MTL) CNN for ODD Prediction.
This is intended to be an improvement over the discrete pipeline based
weathernet architectures due to information sharing through the common backbone.
The model will predict all seven labels from the weathernet++ specification.
"""

from enum import Enum
import torch
import torch.nn as nn
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    efficientnet_b4,
    EfficientNet_B4_Weights,
)
import pickle


class MtlBackbone(Enum):
    """
    Enum for the backbone architecture.
    """

    ResNet50 = "resnet50"
    EfficientNetB4 = "efficientnet_b4"
    Vgg16 = "vgg16"


class MtlWeatherNet(nn.Module):
    """
    WeatherNet: Implementation of the WeatherNet model.
    Requires a backbone to be selected when instantiating the model.
    You can either instantiate it based on a resnet50 or efficientnet_b4 backbone.
    """

    def __init__(self, backbone: MtlBackbone = MtlBackbone.ResNet50) -> None:
        """
        Initialize the MTL WeatherNet model.
        The model is based on ResNet50 architecture.
        """
        super(MtlWeatherNet, self).__init__()
        self.name = "mtl"

        # Shared backbone
        self.backbone_type = backbone
        if backbone == MtlBackbone.ResNet50:
            self.backbone_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.head_in_features = self.backbone_model.fc.in_features
            self.backbone = nn.Sequential(
                *list(self.backbone_model.children())[:-1]
            )  # Remove the classification layer
        elif backbone == MtlBackbone.EfficientNetB4:
            raise ValueError("EfficientNet Backbone not yet implemented")
            # self.backbone = efficientnet_b4(
            #     weights=EfficientNet_B4_Weights.IMAGENET1K_V1
            # )
            # self.backbone = nn.Sequential(
            #     *list(self.backbone.children())[:-1]
            # )  # Remove the classification layer
        elif backbone == MtlBackbone.Vgg16:
            raise ValueError("VGG16 Backbone not yet implemented")
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Task-specific heads. Certain heads are more complex
        # due to the number of classes they predict.

        # Fog prediction head
        self.fog_head = nn.Sequential(
            nn.Conv2d(self.head_in_features, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        # Glare Prediction Head
        self.glare_head = nn.Sequential(
            nn.Conv2d(self.head_in_features, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        # Road prediction
        self.road_head = nn.Sequential(
            nn.Conv2d(self.head_in_features, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )

        # Traffic prediction head
        self.traffic_head = nn.Sequential(
            nn.Conv2d(self.head_in_features, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )

        # Weather prediction head
        self.weather_head = nn.Sequential(
            nn.Conv2d(self.head_in_features, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 6),
        )

        # Scene prediction head
        self.scene_head = nn.Sequential(
            nn.Conv2d(self.head_in_features, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
        )

        # Time of day prediction head
        self.tod_head = nn.Sequential(
            nn.Conv2d(self.head_in_features, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
        )

        # Define loss functions
        # Binary Cross Entropy with Logits Loss for binary classification
        # combines cross entropy with a sigmoid activation function in a single class,
        # making it more numerically stable.
        # For multiclass, we use CrossEntropyLoss (softmax).

        self.fog_loss = nn.BCEWithLogitsLoss()
        self.glare_loss = nn.BCEWithLogitsLoss()

        self.road_loss = nn.CrossEntropyLoss()
        self.traffic_loss = nn.CrossEntropyLoss()
        self.weather_loss = nn.CrossEntropyLoss()
        self.scene_loss = nn.CrossEntropyLoss()
        self.tod_loss = nn.CrossEntropyLoss()

        # model pipelines
        self.num_pipelines = 7

    # TODO: make this an interface thing if possible and use it in all models?
    def get_num_pipelines(self) -> int:
        """
        Get number of pipelines in the model.
        Standard function across all of our custom models.
        Returns:
            int: number of pipelines
        """
        return self.num_pipelines

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Output prediction will be of dimension (batch_size, num_outputs),
        where num_outputs is 22 due to the 22 different labels. This is because
        the multiclass loss functions we use expect raw logits, so we serve multiclass
        predictions as raw logits. In order to turn these into categorical labels, use array
        slicing and torch.argmax as necessary. Binary classification problems are returned
        as a single value (0 or 1) for each class.

        Explained:
            - 1 for fog
            - 1 for glare
            - 3 for road (3 classes)
            - 3 for traffic (3 classes)
            - 6 for weather (6 classes)
            - 4 for scene (4 classes)
            - 4 for time of day (4 classes)
        The total is 1 + 1 + 3 + 3 + 6 + 4 + 4 = 22.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
        Returns:
            predictions: Output tensor of shape (batch_size, num_outputs).
        """
        # Pass through the backbone
        x = self.backbone(x)

        # Pass through the prediction heads
        fog_pred = self.fog_head(x)
        glare_pred = self.glare_head(x)
        road_pred = self.road_head(x)
        traffic_pred = self.traffic_head(x)
        weather_pred = self.weather_head(x)
        scene_pred = self.scene_head(x)
        tod_pred = self.tod_head(x)

        # concat predictions along row axis
        predictions = torch.cat(
            (
                fog_pred,
                glare_pred,
                road_pred,
                traffic_pred,
                weather_pred,
                scene_pred,
                tod_pred,
            ),
            dim=1,
        )
        return predictions

    def compute_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute total loss.
        Args:
            predictions:
            Tuple (fog_pred, glare_pred, road_pred, traffic_pred, weather_pred, scene_pred, tod_pred)
            targets:
            Tuple (fog_target, glare_target, road_target, traffic_target, weather_target, scene_target, tod_target)
        Returns:
            Total loss (scalar)
        """
        # Parse predictions
        fog_pred = predictions[:, 0]
        glare_pred = predictions[:, 1]
        road_pred = predictions[:, 2:5]
        traffic_pred = predictions[:, 5:8]
        weather_pred = predictions[:, 8:14]
        scene_pred = predictions[:, 14:18]
        tod_pred = predictions[:, 18:22]

        # Parse targets
        fog_target = targets[:, 0].float() # float for binary classification tasks
        glare_target = targets[:, 1].float()
        road_target = targets[:, 2]
        traffic_target = targets[:, 3]
        weather_target = targets[:, 4]
        scene_target = targets[:, 5]
        tod_target = targets[:, 6]

        # Compute individual losses
        loss_fog: torch.Tensor = self.fog_loss(fog_pred.squeeze(), fog_target.float())
        loss_glare: torch.Tensor = self.glare_loss(glare_pred.squeeze(), glare_target.float())
        loss_road: torch.Tensor = self.road_loss(road_pred, road_target)
        loss_traffic: torch.Tensor = self.traffic_loss(traffic_pred, traffic_target)
        loss_weather: torch.Tensor = self.weather_loss(weather_pred, weather_target)
        loss_scene: torch.Tensor = self.scene_loss(scene_pred, scene_target)
        loss_tod: torch.Tensor = self.tod_loss(tod_pred, tod_target)

        # Total loss (weighted sum if needed)
        total_loss: torch.Tensor = (
            loss_fog
            + loss_glare
            + loss_road
            + loss_traffic
            + loss_weather
            + loss_scene
            + loss_tod
        )

        # individual loss elements are SCALAR tensors. These are different from floats.
        # return in order seen in README/labels file
        losses = torch.stack(
            [
                loss_fog,
                loss_glare,
                loss_road,
                loss_traffic,
                loss_weather,
                loss_scene,
                loss_tod,
            ]
        )

        # return both summed loss and individual losses
        return total_loss, losses

    def save_checkpoint(self, filepath: str) -> None:
        """
        Save the model's current state to a pickle file.

        Args:
            filepath (str): Path to the file where the model state will be saved.
        """
        with open(filepath, "wb") as f:
            pickle.dump(self.state_dict(), f)

    def load_checkpoint(self, filepath: str) -> None:
        """
        Load the model's state from a pickle file.

        Args:
            filepath (str): Path to the file from which the model state will be loaded.
        """
        with open(filepath, "rb") as f:
            state_dict = pickle.load(f)
        self.load_state_dict(state_dict)

    def get_name(self) -> str:
        """
        Get the name of the model.
        Returns:
            str: Name of the model.
        """
        return "MtlWeatherNet"
