import torch
from torchvision.models import resnet18, resnet50
from LORA import LoRA
import torch.nn as nn


class LoadModel:
    """
    A class to load and modify ResNet models with LoRA (Low-Rank Adaptation) for fine-tuning.

    Attributes:
        model (torch.nn.Module): The selected ResNet model.
        rank (int): Rank parameter for the LoRA layer.
        alpha (float): Scaling factor for the LoRA layer.
    """

    def __init__(self, model_name, num_classes, rank: int = 4, alpha: float = 0.5):
        """
        Initializes the model loader with the specified ResNet model and LoRA parameters.

        Args:
            model_name (str): Name of the ResNet model to load ('resnet18' or 'resnet50').
            num_classes (int): Number of output classes for the classification task.
            rank (int): Rank parameter for LoRA layers.
            alpha (float): Scaling factor for LoRA layers.

        Raises:
            ValueError: If an unsupported model name is provided.
        """
        self.rank = rank
        self.alpha = alpha

        if model_name == "resnet18":
            self.model = resnet18(weights="IMAGENET1K_V1")
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        elif model_name == "resnet50":
            self.model = resnet50(weights="IMAGENET1K_V1")
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        else:
            raise ValueError("Invalid model name. Supported models: resnet18, resnet50")

    def run(self):
        """
        Replaces the fully connected layers in the model with LoRA layers.

        Returns:
            torch.nn.Module: The modified model with LoRA layers.
        """
        # Apply LoRA to fully connected layers in the model
        for name, module in self.model.named_children():
            if isinstance(module, nn.Linear):
                setattr(
                    self.model, name, LoRA(module, alpha=self.alpha, rank=self.rank)
                )

        return self.model
