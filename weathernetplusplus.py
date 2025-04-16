"""
WeatherNet++: Adjusted Implementation of standard WeatherNet model,
augmented to to include scene detection, road detection, and traffic detection.
to conform to BDD100K dataset labels
"""

import pickle
from typing import NamedTuple
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class WeatherNetPlusPlus(nn.Module):
    """WeatherNet: Implementation of the WeatherNet model."""

    class WeatherNetPlusPlusOutput(NamedTuple):
        """
        Structure of the output
        of the forward pass of the model.
        """

        fog_pred: torch.Tensor
        glare_pred: torch.Tensor
        road_pred: torch.Tensor
        traffic_pred: torch.Tensor
        weather_pred: torch.Tensor
        scene_pred: torch.Tensor
        night_pred: torch.Tensor

    def __init__(self) -> None:
        super(WeatherNetPlusPlus, self).__init__()
        self.name = "WeatherNetPlusPlus"

        # fog-net: resnet50, replaced linear layer at end to be ONE output, followed by sigmoid.
        # fog is a single class, so we use a single output with sigmoid activation to predict
        # the probability of fog.
        self.fog_net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.fog_net.fc = nn.Linear(self.fog_net.fc.in_features, 1)

        # glare-net: resnet50, replaced linear layer at end to be ONE output, followed by sigmoid.
        # Glare is a single class, so we use a single output with sigmoid activation to predict
        # the probability of glare.
        self.glare_net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.glare_net.fc = nn.Linear(self.glare_net.fc.in_features, 1)

        # road-net: resnet50, replaced linear layer at end to be THREE outputs, followed by softmax.
        # road is THREE classes (dry road, wet road, snowy road)
        # so we use three outputs with softmax activation to predict the probability of each class.
        self.road_net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.road_net.fc = nn.Linear(self.road_net.fc.in_features, 3)

        # traffic-net: resnet50, replaced linear layer at end to be 3 outputs, followed by softmax.
        # traffic is THREE classes (no traffic, low/moderate traffic, high traffic)
        # so we use three outputs with softmax activation to predict the probability of each class.
        self.traffic_net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.traffic_net.fc = nn.Linear(self.traffic_net.fc.in_features, 3)

        # weather-net: resnet50, replaced linear layer at end to be SIX outputs,
        # followed by softmax.
        # weather is SIX classes (clear, partly cloudy, overcast, rainy, snowy, undefined)
        # so we use six outputs with softmax activation to predict the probability of each class.
        self.weather_net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.weather_net.fc = nn.Linear(self.weather_net.fc.in_features, 6)

        # scene-net: resnet50, replaced linear layer at end to be FOUR outputs, followed by softmax.
        # scene is FOUR classes (residential, highway, city street, undefined),
        # so we use FOUR outputs with softmax activation to predict the probability of each class.
        self.scene_net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.scene_net.fc = nn.Linear(self.scene_net.fc.in_features, 4)

        # night-net: resnet50, replaced linear layer at end to be FOUR output, followed by softmax.
        # timeofday is FOUR classes (dawn/dusk, daytime, night, undefined), so we use four output
        # neurons with softmax activation to predict the probability of each class.
        self.night_net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.night_net.fc = nn.Linear(self.night_net.fc.in_features, 4)

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
        self.night_loss = nn.CrossEntropyLoss()

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

    def forward(self, x):
        """
        forward pass
        gets called by model()
        returns a Tuple:
        (fog_pred, glare_pred, road_pred, traffic_pred, weather_pred, scene_pred, night_pred)
        """
        # fog-net prediction, 1 class
        fog = self.fog_net(x)

        # glare-net prediction, 1 class
        glare = self.glare_net(x)

        # road-net prediction, 3 classes
        road = self.road_net(x)

        # traffic-net prediction, 3 class
        traffic = self.traffic_net(x)

        # weather-net prediction, 6 classes
        weather = self.weather_net(x)

        # scene-net prediction, 4 classes
        scene = self.scene_net(x)

        # night-net prediction, 4 classes
        night = self.night_net(x)

        # return predictions in order seen in labels file
        return torch.cat((fog, glare, road, traffic, weather, scene, night), dim=1)

    def compute_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        fog_pred = predictions[:, 0]
        glare_pred = predictions[:, 1]
        road_pred = predictions[:, 2:5]
        traffic_pred = predictions[:, 5:8]
        weather_pred = predictions[:, 8:14]
        scene_pred = predictions[:, 14:18]
        night_pred = predictions[:, 18:22]

        # Parse targets
        fog_target = targets[:, 0].float() # float for binary classification tasks
        glare_target = targets[:, 1].float()
        road_target = targets[:, 2]
        traffic_target = targets[:, 3]
        weather_target = targets[:, 4]
        scene_target = targets[:, 5]
        night_target = targets[:, 6]

        # Compute individual losses

        # BCEWithLogitsLoss expects (batch,) logits and (batch,) labels
        loss_fog: torch.Tensor = self.fog_loss(fog_pred.squeeze(), fog_target.float())

        # BCEWithLogitsLoss expects (batch,) logits and (batch,) labels
        loss_glare: torch.Tensor = self.glare_loss(glare_pred.squeeze(), glare_target.float())

        # CrossEntropyLoss expects (batch, 3) logits and (batch,) labels
        loss_road: torch.Tensor = self.road_loss(road_pred, road_target)

        # CrossEntropyLoss expects (batch, 3) logits and (batch,) labels
        loss_traffic: torch.Tensor = self.traffic_loss(traffic_pred, traffic_target)

        # CrossEntropyLoss expects (batch, 6) logits and (batch,) labels
        loss_weather: torch.Tensor = self.weather_loss(weather_pred, weather_target)

        # CrossEntropyLoss expects (batch, 4) logits and (batch,) labels
        loss_scene: torch.Tensor = self.scene_loss(scene_pred, scene_target)

        # CrossEntropyLoss expects (batch, 4) logits and (batch,) labels
        loss_night: torch.Tensor = self.night_loss(night_pred, night_target)

        # Total loss (weighted sum if needed)
        total_loss: torch.Tensor = (
            loss_fog
            + loss_glare
            + loss_road
            + loss_traffic
            + loss_weather
            + loss_scene
            + loss_night
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
                loss_night,
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
        return "WeatherNetPlusPlus"
