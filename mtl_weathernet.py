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
        self.name = "MTL"

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
            nn.Linear(self.head_in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        # Glare Prediction Head
        self.glare_head = nn.Sequential(
            nn.Linear(self.head_in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        # Road prediction
        self.road_head = nn.Sequential(
            nn.Linear(self.head_in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )

        # Traffic prediction head
        self.traffic_head = nn.Sequential(
            nn.Linear(self.head_in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )

        # Weather prediction head
        self.weather_head = nn.Sequential(
            nn.Linear(self.head_in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 6),
        )

        # Scene prediction head
        self.scene_head = nn.Sequential(
            nn.Linear(self.head_in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
        )

        # Time of day prediction head
        self.tod_head = nn.Sequential(
            nn.Linear(self.head_in_features, 512),
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

    def compute_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor]:
        """
        Compute total loss.
        Args:
            predictions:
            Tuple (fog_pred, glare_pred, road_pred, traffic_pred, weather_pred, scene_pred, night_pred)
            targets:
            Tuple (fog_target, glare_target, road_target, traffic_target, weather_target, scene_target, night_target)
        Returns:
            Total loss (scalar)
        """
        # Parse predictions
        night_pred = predictions[:, 0:4]
        glare_pred = predictions[:, 4]
        weather_pred = predictions[:, 5:11]
        fog_pred = predictions[:, 11]
        road_pred = predictions[:, 12:15]
        traffic_pred = predictions[:, 15:18]
        scene_pred = predictions[:, 18:22]

        # Parse targets
        night_target = targets[:, 0]
        glare_target = targets[:, 1].float()  # float for binary classification tasks
        weather_target = targets[:, 2]
        fog_target = targets[:, 3].float()
        road_target = targets[:, 4]
        traffic_target = targets[:, 5]
        scene_target = targets[:, 6]

        # Compute individual losses

        # CrossEntropyLoss expects (batch, 4) logits and (batch,) labels
        loss_night = self.tod_loss(night_pred, night_target)

        # BCEWithLogitsLoss expects (batch,) logits and (batch,) labels
        loss_glare = self.glare_loss(glare_pred.squeeze(), glare_target.float())

        # CrossEntropyLoss expects (batch, 6) logits and (batch,) labels
        loss_weather = self.weather_loss(weather_pred, weather_target)

        # BCEWithLogitsLoss expects (batch,) logits and (batch,) labels
        loss_fog = self.fog_loss(fog_pred.squeeze(), fog_target.float())

        # CrossEntropyLoss expects (batch, 3) logits and (batch,) labels
        loss_road = self.road_loss(road_pred, road_target)

        # CrossEntropyLoss expects (batch, 3) logits and (batch,) labels
        loss_traffic = self.traffic_loss(traffic_pred, traffic_target)

        # CrossEntropyLoss expects (batch, 4) logits and (batch,) labels
        loss_scene = self.scene_loss(scene_pred, scene_target)

        # Total loss (weighted sum if needed)
        total_loss = (
            loss_night
            + loss_glare
            + loss_weather
            + loss_fog
            + loss_road
            + loss_traffic
            + loss_scene
        )

        # individual loss elements are SCALAR tensors. These are different from floats.
        # return in order seen in labels file
        losses = torch.stack(
            [
                loss_night,
                loss_glare,
                loss_weather,
                loss_fog,
                loss_road,
                loss_traffic,
                loss_scene,
            ]
        )

        # return both summed loss and individual losses
        return total_loss, losses
