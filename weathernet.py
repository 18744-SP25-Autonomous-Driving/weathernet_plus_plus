"""
WeatherNet: Standard WeatherNet model
working with Bdd100k labels. Deals with 4 labels.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import pickle


class WeatherNet(nn.Module):
    """WeatherNet: Implementation of the WeatherNet model."""

    def __init__(self) -> None:
        super(WeatherNet, self).__init__()

        # night-net: resnet50, replaced linear layer at end to be FOUR output, followed by softmax.
        # Time of day is 4 classes (dawn/dusk, daytime, night, undefined), so we use four output
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

        # model pipelines
        self.num_pipelines = 4

    # TODO: make this an interface thing if possible and use it in all models?
    def get_num_pipelines(self) -> int:
        """
        Get number of pipelines in the model.
        Standard function across all of our custom models.
        Returns:
            int: number of pipelines
        """
        return self.num_pipelines

    def forward(self, x) -> torch.Tensor:
        """
        forward pass
        gets called by model()
        returns a Torch.Tensor of shape (batch_size, num_outputs)
        where num_outputs is 12 due to the 12 individual classes.
        (ie. function is returning raw logits for each multiclass classifer)
        Args:
            x: Torch.Tensor (batch_size, 3, 224, 224)
        Returns:
            Torch.Tensor (batch_size, num_outputs)
        """
        # night-net prediction, 4 classes
        night = self.night_net(x)

        # glare-net prediction, 1 class
        glare = self.glare_net(x)

        # weather-net prediction, 6 classes
        weather = self.weather_net(x)

        # fog-net prediction, 1 class
        fog = self.fog_net(x)

        # return predictions
        return torch.cat((night, glare, weather, fog), dim=1)

    def compute_loss(self, predictions, targets) -> tuple[torch.Tensor,torch.Tensor]:
        """
        Compute total loss. Requires the Predictions generated
        by the `forward` function, as well as the input labels.
        The reason these are of different dimension is because the `forward`
        function returns the raw logits for each multiclass classifier.
        The loss function expects the labels to be in the categorical format.
        Args:
            predictions: Torch.Tensor (night_pred, glare_pred, weather_pred, fog_pred)
                1. night_pred: Torch.Tensor (batch_size, 4)
                2. glare_pred: Torch.Tensor (batch_size, 1)
                3. weather_pred: Torch.Tensor (batch_size, 6)
                4. fog_pred: Torch.Tensor (batch_size, 1)
            targets: Torch.Tensor (night_target, glare_target, weather_target, fog_target)
                1. night_target: Torch.Tensor (batch_size, 1)
                2. glare_target: Torch.Tensor (batch_size, 1)
                3. weather_target: Torch.Tensor (batch_size, 1)
                4. fog_target: Torch.Tensor (batch_size, 1)
        Returns:
            Total loss (scalar)
        """

        night_pred = predictions[:, 0:4]
        glare_pred = predictions[:, 4]
        weather_pred = predictions[:, 5:11]
        fog_pred = predictions[:, 11]

        # CrossEntropy expects label indices (ints/longs)
        # BCEWithLogits expects label probabilities (floats)
        night_target = targets[:, 0]
        glare_target = targets[:, 1].float()
        weather_target = targets[:, 2]
        fog_target = targets[:, 3].float()

        # Compute individual losses

        # CrossEntropyLoss expects (batch, 4) logits and (batch,) labels
        # print(night_pred.shape, night_target.shape)
        loss_night: torch.Tensor = self.night_loss(night_pred, night_target)

        # BCEWithLogitsLoss expects (batch,) logits and (batch,) labels
        loss_glare: torch.Tensor = self.glare_loss(glare_pred.squeeze(), glare_target)

        # CrossEntropyLoss expects (batch, 5) logits and (batch,) labels
        loss_weather: torch.Tensor = self.weather_loss(weather_pred, weather_target)

        # BCEWithLogitsLoss expects (batch,) logits and (batch,) labels
        loss_fog: torch.Tensor = self.fog_loss(fog_pred.squeeze(), fog_target)

        # Total loss (weighted sum if needed)
        total_loss: torch.Tensor = loss_night + loss_glare + loss_weather + loss_fog

        # individual loss elements are SCALAR tensors. These are different from floats.
        # tensors have unique properties, like .device, .requires_grad (if True, tracks gradients for backprop)
        # and methods like .backward() to do backprop, or .item() to get the value as a python float.
        losses = torch.stack([loss_night, loss_glare, loss_weather, loss_fog])

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
        return "MtlWeatherNetPlusPlus"
