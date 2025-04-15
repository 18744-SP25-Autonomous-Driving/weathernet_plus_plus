"""
WeatherNet Transformer: Implementation of the WeatherNet model using a Vision Transformer backbone.
This model predicts all seven labels from the WeatherNet++ specification.
"""

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
import pickle


class WeatherNetTransformer(nn.Module):
    """
    WeatherNet Transformer: Implementation of the WeatherNet model using a Vision Transformer backbone.
    """

    def __init__(self) -> None:
        """
        Initialize the WeatherNet Transformer model.
        The model is based on a Vision Transformer (ViT) architecture.
        """
        super(WeatherNetTransformer, self).__init__()
        self.name = "WeatherNetTransformer"

        # Shared Vision Transformer backbone
        self.backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.head_in_features = self.backbone.heads.head.in_features
        self.backbone.heads = nn.Identity()  # Remove the classification head

        # Task-specific heads

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
        self.fog_loss = nn.BCEWithLogitsLoss()
        self.glare_loss = nn.BCEWithLogitsLoss()
        self.road_loss = nn.CrossEntropyLoss()
        self.traffic_loss = nn.CrossEntropyLoss()
        self.weather_loss = nn.CrossEntropyLoss()
        self.scene_loss = nn.CrossEntropyLoss()
        self.tod_loss = nn.CrossEntropyLoss()

        # model pipelines
        self.num_pipelines = 7

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
        where num_outputs is 22 due to the 22 different labels.

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
        loss_night: torch.Tensor = self.tod_loss(night_pred, night_target)

        # BCEWithLogitsLoss expects (batch,) logits and (batch,) labels
        loss_glare: torch.Tensor = self.glare_loss(
            glare_pred.squeeze(), glare_target.float()
        )

        # CrossEntropyLoss expects (batch, 6) logits and (batch,) labels
        loss_weather: torch.Tensor = self.weather_loss(weather_pred, weather_target)

        # BCEWithLogitsLoss expects (batch,) logits and (batch,) labels
        loss_fog: torch.Tensor = self.fog_loss(fog_pred.squeeze(), fog_target.float())

        # CrossEntropyLoss expects (batch, 3) logits and (batch,) labels
        loss_road: torch.Tensor = self.road_loss(road_pred, road_target)

        # CrossEntropyLoss expects (batch, 3) logits and (batch,) labels
        loss_traffic: torch.Tensor = self.traffic_loss(traffic_pred, traffic_target)

        # CrossEntropyLoss expects (batch, 4) logits and (batch,) labels
        loss_scene: torch.Tensor = self.scene_loss(scene_pred, scene_target)

        # Total loss (weighted sum if needed)
        total_loss: torch.Tensor = (
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
        losses: torch.Tensor = torch.stack(
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
        return "WeatherNetTransformer"
