"""
WeatherNet++: Adjusted Implementation of standard WeatherNet model,
augmented to to include scene detection, road detection, and traffic detection.
to conform to BDD100K dataset labels
"""

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
        # night-net prediction, 4 classes
        night = self.night_net(x)

        # glare-net prediction, 1 class
        glare = self.glare_net(x)

        # weather-net prediction, 6 classes
        weather = self.weather_net(x)

        # fog-net prediction, 1 class
        fog = self.fog_net(x)

        # road-net prediction, 3 classes
        road = self.road_net(x)

        # traffic-net prediction, 3 class
        traffic = self.traffic_net(x)

        # scene-net prediction, 4 classes
        scene = self.scene_net(x)

        # return predictions in order seen in labels file
        return torch.cat((night, glare, weather, fog, road, traffic, scene), dim=1)
    
        # return WeatherNetPlusPlus.WeatherNetPlusPlusOutput(fog, glare, road, traffic, weather, scene, night)

    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor]:
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
        glare_target = targets[:, 1].float() # float for binary classification tasks
        weather_target = targets[:, 2]
        fog_target = targets[:, 3].float()
        road_target = targets[:, 4]
        traffic_target = targets[:, 5]
        scene_target = targets[:, 6]

        # Compute individual losses

        # CrossEntropyLoss expects (batch, 4) logits and (batch,) labels
        loss_night = self.night_loss(night_pred, night_target)
        
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
        losses = torch.stack([loss_night, loss_glare, loss_weather, loss_fog, loss_road, loss_traffic, loss_scene])

        # return both summed loss and individual losses
        return total_loss, losses
