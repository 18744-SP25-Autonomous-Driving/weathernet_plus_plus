'''WeatherNet: Implementation of the standard WeatherNet model.'''
from typing import List, Tuple, TypedDict
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class WeatherNet(nn.Module):
    '''WeatherNet: Implementation of the WeatherNet model.'''
    class WeatherNetLoss():
        '''WeathernetLoss: Wrapper for Weathernet pipeline-wise loss objects, and total loss.'''
        def __init__(self, loss_dict: dict) -> None:
            self.night_loss:    int =   loss_dict['night_loss']
            self.glare_loss:    int =   loss_dict['glare_loss']
            self.fog_loss:      int =   loss_dict['fog_loss']
            self.precip_loss:   int =   loss_dict['precip_loss']
            self.total_loss:    int =   loss_dict['total_loss']

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

        # precipitation-net: resnet50, replaced linear layer at end to be THREE output, followed
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
        # Precipitation is multiclass, so we use CrossEntropyLoss (softmax).
        self.night_loss = nn.BCEWithLogitsLoss()
        self.glare_loss = nn.BCEWithLogitsLoss()
        self.fog_loss = nn.BCEWithLogitsLoss()
        self.precipitation_loss = nn.CrossEntropyLoss()


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

    def backward(self):
        '''backward pass'''
        self.night_loss().backward()
        self.glare_loss().backward()
        self.fog_loss().backward()
        self.precipitation_loss().backward()

    def loss(self, predictions, labels)->WeatherNetLoss:
        '''
        Compute and return loss of model outputs.
        '''
        night, glare, precipitation, fog = predictions.split(1, 1, 3, 1, dim=1)

        # predictions will be in a 2d, 1 column tensor, so we need to unsqueeze the labels to match
        night_loss_val  = self.night_loss(night, labels[:, 0].unsqueeze(1))
        glare_loss_val  = self.glare_loss(glare, labels[:, 1].unsqueeze(1))
        fog_loss_val    = self.fog_loss(fog, labels[:, 2].unsqueeze(1))
        precipitation_loss_val = self.precipitation_loss(precipitation, labels[:, 3].long())

        return WeatherNet.WeatherNetLoss({
            'night_loss': night_loss_val,
            'glare_loss': glare_loss_val,
            'fog_loss': fog_loss_val,
            'precip_loss': precipitation_loss_val,
            'total_loss': night_loss_val + glare_loss_val + fog_loss_val + precipitation_loss_val
        })
