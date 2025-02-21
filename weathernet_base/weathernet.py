'''WeatherNet: Implementation of the WeatherNet model.'''
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class WeatherNet(nn.Module):
    '''WeatherNet: Implementation of the WeatherNet model.'''
    def __init__(self) -> None:
        super(WeatherNet, self).__init__()

        # night-net: resnet50, replaced linear layer at end to be one output, followed by sigmoid.
        # Time of day is a single class, so we use a single output with sigmoid activation 
        # to predict the probability of night.
        self.night_net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.night_net.fc = nn.Linear(self.night_net.fc.in_features, 1)

        # glare-net: resnet50, replaced linear layer at end to be one output, followed by sigmoid.
        # Glare is a single class, so we use a single output with sigmoid activation to predict
        # the probability of glare.
        self.glare_net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.glare_net.fc = nn.Linear(self.glare_net.fc.in_features, 1)

        # precipitation-net: resnet50, replaced linear layer at end to be TWO output, followed
        # by softmax.Precipitation is three classes (rain, snow, clear), so we use three outputs
        # with softmax activation to predict the probability of each class. First element is rain,
        # second is snow.
        self.precipitation_net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.precipitation_net.fc = nn.Linear(self.precipitation_net.fc.in_features, 3)

        # fog-net: resnet50, replaced linear layer at end to be one output, followed by sigmoid.
        # Fog is a single class, so we use a single output with sigmoid activation to predict
        # the probability of fog.
        self.fog_net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.fog_net.fc = nn.Linear(self.fog_net.fc.in_features, 1)

        # Define loss functions
        # Binary Cross Entropy with Logits Loss for binary classification
        # combines cross entropy with a sigmoid activation function in a single class,
        # making it more numerically stable.
        # We use this for night, glare, and fog.
        self.binary_criterion = nn.BCEWithLogitsLoss()

        # Softmax is internally computed. We use this for precipitation.
        self.multi_criterion = nn.CrossEntropyLoss()


    def forward(self,x):
        '''forward pass'''
        # night-net prediction
        night = self.night_net(x)

        # glare-net prediction
        glare = self.glare_net(x)

        # precipitation-net prediction
        precipitation = self.precipitation_net(x)

        # fog-net prediction
        fog = self.fog_net(x)

        return torch.cat([night, glare, precipitation, fog], dim=1)

    def criterion(self, predictions, labels):
        '''compute loss of model outputs'''
        night, glare, precipitation, fog = predictions.split(1, 1, 3, 1, dim=1)

        night_loss  = self.binary_criterion(night, labels[:, 0].unsqueeze(1))
        glare_loss  = self.binary_criterion(glare, labels[:, 1].unsqueeze(1))
        fog_loss    = self.binary_criterion(fog, labels[:, 2].unsqueeze(1))
        precipitation_loss = self.multi_criterion(precipitation, labels[:, 3].long())

        total_loss = night_loss + glare_loss + fog_loss + precipitation_loss
        return total_loss, night_loss, glare_loss, fog_loss, precipitation_loss

    def compute_metrics(self, predictions, labels):
        '''compute metrics of model outputs'''
        #TODO
        return
