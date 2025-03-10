"""
WeatherNet++: Adjusted Implementation of standard WeatherNet model,
augmented to to include scene detection, road detection, and traffic detection.
to conform to BDD100K dataset labels
"""

from typing import List, Tuple, TypedDict
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class WeatherNetPlusPlus(nn.Module):
    """WeatherNet: Implementation of the WeatherNet model."""

    def __init__(self) -> None:
        super(WeatherNetPlusPlus, self).__init__()

        # night-net: resnet50, replaced linear layer at end to be FOUR output, followed by softmax.
        # Time of day is 4 classes (dawn/dusk, daytime, night, undefined), so we use a four output
        # neurons with softmax activation to predict the probability of each one.
        self.night_net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.night_net.fc = nn.Linear(self.night_net.fc.in_features, 4)

        # weather-net: resnet50, replaced linear layer at end to be FIVE output,
        # followed by softmax.
        # Weather is five classes (clear, rain, snow, partly cloudy, undefined),
        # so we use five outputs with
        # softmax activation to predict the probability of each class.
        self.weather_net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.weather_net.fc = nn.Linear(self.weather_net.fc.in_features, 5)

        # scene-net: resnet50, replaced linear layer at end to be SIX output, followed by softmax.
        # Scene is six classes ('city street', 'highway', 'residential', 'undefined', 'parking lot', 'tunnel'),
        # so we use six outputs with softmax activation to predict the probability of each class.
        self.scene_net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.scene_net.fc = nn.Linear(self.scene_net.fc.in_features, 6)

        # glare-net: resnet50, replaced linear layer at end to be ONE output, followed by sigmoid.
        # Glare is a single class, so we use a single output with sigmoid activation to predict
        # the probability of glare.
        self.glare_net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.glare_net.fc = nn.Linear(self.glare_net.fc.in_features, 1)

        # road-net: resnet50, replaced linear layer at end to be ONE output, followed by softmax.
        # Road is single class, so we use one output with sigmoid activation to predict
        # the probability of it being road
        self.road_net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.road_net.fc = nn.Linear(self.road_net.fc.in_features, 1)

        # traffic-net: resnet50, replaced linear layer at end to be ONE output, followed by softmax.
        # Traffic is single class, so we use one output with sigmoid activation to predict
        # the probability of it being traffic
        self.traffic_net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.traffic_net.fc = nn.Linear(self.traffic_net.fc.in_features, 1)

        # Define loss functions
        # Binary Cross Entropy with Logits Loss for binary classification
        # combines cross entropy with a sigmoid activation function in a single class,
        # making it more numerically stable.
        # For multiclass, we use CrossEntropyLoss (softmax).

        self.night_loss = nn.CrossEntropyLoss()
        self.weather_loss = nn.CrossEntropyLoss()
        self.scene_loss = nn.CrossEntropyLoss()

        self.glare_loss = nn.BCEWithLogitsLoss()
        self.road_loss = nn.BCEWithLogitsLoss()
        self.traffic_loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        """
        forward pass
        gets called by model()
        returns a Tuple (night_pred, glare_pred, weather_pred)
        """
        # night-net prediction, 4 classes
        night = self.night_net(x)

        # glare-net prediction, 1 class
        glare = self.glare_net(x)

        # weather-net prediction, 5 classes
        weather = self.weather_net(x)

        # scene-net prediction, 6 classes
        scene = self.scene_net(x)

        # road-net prediction, 1 class
        road = self.road_net(x)

        # traffic-net prediction, 1 class
        traffic = self.traffic_net(x)

        return night, glare, weather, scene, road, traffic

    def compute_loss(self, predictions, targets):
        """
        Compute total loss.
        Args:
            predictions: Tuple (night_pred, glare_pred, weather_pred, scene_pred, road_pred, traffic_pred)
            targets: Tuple (night_target, glare_target, weather_target, scene_target, road_target, traffic_target)
        Returns:
            Total loss (scalar)
        """
        (
            night_pred,
            glare_pred,
            weather_pred,
            scene_pred,
            road_pred,
            traffic_pred
        ) = predictions
        
        (
            night_target,
            glare_target,
            weather_target,
            scene_target,
            road_target,
            traffic_target,
        ) = targets

        # Compute individual losses

        # CrossEntropyLoss expects (batch, 4) logits and (batch,) labels
        loss_night = self.night_loss(night_pred, night_target)

        # BCEWithLogitsLoss expects (batch,) logits and (batch,) labels
        loss_glare = self.glare_loss(glare_pred.squeeze(), glare_target.float())

        # CrossEntropyLoss expects (batch, 5) logits and (batch,) labels
        loss_weather = self.weather_loss(weather_pred, weather_target)

        # CrossEntropyLoss expects (batch, 6) logits and (batch,) labels
        loss_scene = self.scene_loss(scene_pred, scene_target)

        # BCEWithLogitsLoss expects (batch,) logits and (batch,) labels
        loss_road = self.road_loss(road_pred.squeeze(), road_target.float())

        # BCEWithLogitsLoss expects (batch,) logits and (batch,) labels
        loss_traffic = self.traffic_loss(traffic_pred.squeeze(), traffic_target.float())

        # Total loss (weighted sum if needed)
        total_loss = (
            loss_night
            + loss_glare
            + loss_weather
            + loss_scene
            + loss_road
            + loss_traffic
        )
        return total_loss
