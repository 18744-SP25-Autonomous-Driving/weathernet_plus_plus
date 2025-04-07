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

        # Shared backbone
        self.backbone_type = backbone
        if backbone == MtlBackbone.ResNet50:
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.backbone = nn.Sequential(
                *list(self.backbone.children())[:-1]
            )  # Remove the classification layer
        elif backbone == MtlBackbone.EfficientNetB4:
            self.backbone = efficientnet_b4(
                weights=EfficientNet_B4_Weights.IMAGENET1K_V1
            )
            self.backbone = nn.Sequential(
                *list(self.backbone.children())[:-1]
            )  # Remove the classification layer
        elif backbone == MtlBackbone.Vgg16:
            raise ValueError("VGG16 Backbone not yet implemented")
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Task-specific heads. Certain heads are more complex
        # due to the number of classes they predict.

        # Fog prediction head
        self.fog_head = nn.Sequential(
            nn.Linear(self.backbone.fc_in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        # Glare Prediction Head
        self.glare_head = nn.Sequential(
            nn.Linear(self.backbone.fc_in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        # Road prediction
        self.road_head = nn.Sequential(
            nn.Linear(self.backbone.fc_in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )

        # Traffic prediction head
        self.traffic_head = nn.Sequential(
            nn.Linear(self.backbone.fc_in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )

        # Weather prediction head
        self.weather_head = nn.Sequential(
            nn.Linear(self.backbone.fc_in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 5),
        )

        # Scene prediction head
        self.scene_head = nn.Sequential(
            nn.Linear(self.backbone.fc_in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )

        # Time of day prediction head
        self.traffic_head = nn.Sequential(
            nn.Linear(self.backbone.fc_in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
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
        self.pipelines = ["fog", "glare", "road", "traffic", "weather", "scene", "night"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Output prediction will be of dimension (batch_size, num_outputs),
        where num_outputs is 19 due to the 19 different labels. This is because
        the multiclass loss functions we use expect raw logits, so we serve multiclass 
        predictions as raw logits. In order to turn these into categorical labels, use array
        slicing and torch.argmax as necessary. Binary classification problems are returned
        as a single value (0 or 1) for each class.

        Explained:
            - 1 for fog
            - 1 for glare
            - 3 for road (3 classes)
            - 3 for traffic (3 classes)
            - 5 for weather (5 classes)
            - 3 for scene (4 classes)
            - 3 for time of day (3 classes)
        The total is 1 + 1 + 3 + 3 + 5 + 3 + 3 = 19.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
        Returns:
            predictions: Output tensor of shape (batch_size, num_outputs).
        """
        # Pass through the backbone
        x = self.backbone(x)

        # Adaptive average pooling to make the feature map size consistent
        x = torch.flatten(x, 1)

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

    def compute_loss(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss for the model.
        Args:
            preds: Predictions from the model. Shape (batch_size, num_outputs).
            labels: Ground truth labels. Shape (batch_size, num_outputs).
        Returns:
            loss: Computed loss
        """
        fog_loss = self.fog_loss(preds[:, 0], labels[:, 0])  # fog is at index 0

        glare_loss = self.glare_loss(preds[:, 1], labels[:, 1])  # glare is at index 1

        road_loss = self.road_loss(
            preds[:, 2:5], labels[:, 2:5].long()
        )  # road is at index 2-4

        traffic_loss = self.traffic_loss(
            preds[:, 5:8], labels[:, 5:8].long()
        )  # traffic is at index 5-7

        weather_loss = self.weather_loss(
            preds[:, 8:13], labels[:, 8:13].long()
        )  # weather is at index 8-12

        scene_loss = self.scene_loss(
            preds[:, 13:16], labels[:, 13:16].long()
        )  # scene is at index 13-15

        tod_loss = self.tod_loss(
            preds[:, 16:], labels[:, 16:].long()
        )  # timeofday is at index 16-18

        # for now, assume all losses
        # are equally weighted
        total_loss = (
            fog_loss
            + glare_loss
            + road_loss
            + traffic_loss
            + weather_loss
            + scene_loss
            + tod_loss
        )
        return total_loss
