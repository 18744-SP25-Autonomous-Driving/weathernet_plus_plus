'''
Base Model Class for all the models in the project.
'''


from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    """
    Base model class for all models in the project.
    This class inherits from nn.Module and ABC (Abstract Base Class).
    It provides a common interface for all models.
    """
    def __init__(self) -> None:
        """
        Initializes the BaseModel class.
        Calls the constructor of the parent class (nn.Module).
        """
        super(BaseModel, self).__init__()


    @abstractmethod
    def get_num_pipelines(self) -> int:
        """
        Abstract method to get the number of pipelines in the model.
        This method must be implemented by all subclasses.
        
        :return: Number of pipelines in the model (int).
        """
        raise NotImplementedError("Subclasses must implement this method.")


    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Abstract method for the forward pass of the model.
        This method must be implemented by all subclasses.
        
        :param input: Input tensor(s) for the model.
        :return: Output tensor(s) from the model.
        """
        raise NotImplementedError("Subclasses must implement this method.")


    @abstractmethod
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:   
        """
        Abstract method to compute the loss for the model.
        This method must be implemented by all subclasses.
        
        :param predictions: The output predictions from the model.
        :param targets: The ground truth labels for the input data.
        :return: Computed loss (scalar).
        """
        raise NotImplementedError("Subclasses must implement this method.")