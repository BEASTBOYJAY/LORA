import torch
from torchvision.models import resnet18, resnet50
from LORA import LoRA
import torch.nn as nn


class LoadModel:
    def __init__(
        self,
        model_name,
        num_classes,
        rank: int = 4,
        alpha: float = 0.5,
    ):
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
        for name, module in self.model.named_children():
            if isinstance(module, nn.Linear):
                setattr(
                    self.model, name, LoRA(module, alpha=self.alpha, rank=self.rank)
                )

        return self.model
