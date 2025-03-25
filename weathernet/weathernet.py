"""
WeatherNet: Adjusted Implementation of standard WeatherNet model 
to conform to BDD100K dataset labels
"""

from typing import List, Tuple, TypedDict, NamedTuple
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class WeatherNet(nn.Module):
    """WeatherNet: Implementation of the WeatherNet model."""

    class WeatherNetOutput(NamedTuple):
        '''
        Structure of the output
        of the forward pass of the model.
        '''
        night_pred: torch.Tensor
        glare_pred: torch.Tensor
        weather_pred: torch.Tensor
        fog_pred: torch.Tensor

    def __init__(self) -> None:
        super(WeatherNet, self).__init__()

        # night-net: resnet50, replaced linear layer at end to be FOUR output, followed by softmax.
        # Time of day is 4 classes (dawn/dusk, daytime, night, undefined), so we use a four output
        # neurons with softmax activation to predict the probability of each one.
        self.night_net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.night_net.fc = nn.Linear(self.night_net.fc.in_features, 4)

        # glare-net: resnet50, replaced linear layer at end to be one output, followed by sigmoid.
        # Glare is a single class, so we use a single output with sigmoid activation to predict
        # the probability of glare.
        self.glare_net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.glare_net.fc = nn.Linear(self.glare_net.fc.in_features, 1)

        # weather-net: resnet50, replaced linear layer at end to be FIVE output,
        # followed by softmax.
        # Weather is SIX classes (clear, partly cloudy, overcast, rainy, snowy, undefined)
        # so we use five outputs with
        # softmax activation to predict the probability of each class.
        # This replaces the original WeatherNet's Precipitation Classifier.
        self.weather_net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.weather_net.fc = nn.Linear(self.weather_net.fc.in_features, 6)

        # fog-net: resnet50, replaced linear layer at end to be one output, followed by sigmoid.
        # Fog is a single class, so we use a single output with sigmoid activation to predict
        # the probability of fog.
        self.fog_net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.fog_net.fc = nn.Linear(self.fog_net.fc.in_features, 1)
        # Define loss functions
        # Binary Cross Entropy with Logits Loss for binary classification
        # combines cross entropy with a sigmoid activation function in a single class,
        # making it more numerically stable.
        # For multiclass, we use CrossEntropyLoss (softmax).

        self.night_loss = nn.CrossEntropyLoss()
        self.glare_loss = nn.BCEWithLogitsLoss()
        self.weather_loss = nn.CrossEntropyLoss()
        self.fog_loss = nn.BCEWithLogitsLoss()


    def forward(self, x) -> WeatherNetOutput:
        """
        forward pass
        gets called by model()
        returns a Tuple (night_pred, glare_pred, weather_pred) 
        """
        # night-net prediction, 4 classes
        night = self.night_net(x)

        # glare-net prediction, 1 class
        glare = self.glare_net(x)

        # weather-net prediction, 6 classes
        weather = self.weather_net(x)

        # fog-net prediction, 1 class
        fog = self.fog_net(x)

        return WeatherNet.WeatherNetOutput(night, glare, weather, fog)

    def compute_loss(self, predictions, targets):
        """
        Compute total loss.
        Args:
            predictions: Tuple (night_pred, glare_pred, weather_pred, fog_pred)
            targets: Tuple (night_target, glare_target, weather_target, fog_target)
        Returns:
            Total loss (scalar)
        """
        night_pred, glare_pred, weather_pred, fog_pred = predictions
        night_target, glare_target, weather_target, fog_target = targets

        # Compute individual losses

        # CrossEntropyLoss expects (batch, 4) logits and (batch,) labels
        loss_night = self.night_loss(night_pred, night_target)

        # BCEWithLogitsLoss expects (batch,) logits and (batch,) labels
        loss_glare = self.glare_loss(glare_pred.squeeze(), glare_target.float())

        # CrossEntropyLoss expects (batch, 5) logits and (batch,) labels
        loss_weather = self.weather_loss(weather_pred, weather_target)

        # BCEWithLogitsLoss expects (batch,) logits and (batch,) labels
        loss_fog = self.fog_loss(fog_pred.squeeze(), fog_target.float())

        # Total loss (weighted sum if needed)
        total_loss = loss_night + loss_glare + loss_weather + loss_fog
        return total_loss
