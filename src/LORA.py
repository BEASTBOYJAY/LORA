import torch
import torch.nn as nn


class LoRA(nn.Module):
    """
    LoRA (Low-Rank Adaptation) module for adapting existing layers with efficient parameterization.

    This module wraps a given layer and adds learnable low-rank adaptation matrices.

    Attributes:
        original_layer (nn.Module): The original fully connected layer.
        rank (int): The rank of the low-rank adaptation.
        lora_A (torch.nn.Parameter): Learnable parameter matrix A.
        lora_B (torch.nn.Parameter): Learnable parameter matrix B.
        use_lora (bool): Flag to enable or disable LoRA during forward propagation.
    """

    def __init__(self, original_layer, alpha: float = 0.5, rank: int = 4):
        """
        Initializes the LoRA module.

        Args:
            original_layer (nn.Module): The original layer to be wrapped by LoRA.
            alpha (float): Scaling factor for the LoRA matrices.
            rank (int): Rank of the low-rank adaptation.
        """
        super(LoRA, self).__init__()
        self.original_layer = original_layer
        self.rank = rank

        # Initialize LoRA matrices with scaling by alpha
        self.lora_A = nn.Parameter(
            torch.randn(rank, original_layer.in_features) * alpha
        )
        self.lora_B = nn.Parameter(
            torch.randn(original_layer.out_features, rank) * alpha
        )

        self.use_lora = True

    def forward(self, x):
        """
        Forward pass through the LoRA-enhanced layer.

        Args:
            x (torch.Tensor): Input tensor to the layer.

        Returns:
            torch.Tensor: The output after applying LoRA and the original layer.
        """
        if self.use_lora:
            # Apply the original layer and the low-rank adaptation
            return self.original_layer(x) + torch.matmul(
                torch.matmul(x, self.lora_A.T), self.lora_B.T
            )
        else:
            # Apply only the original layer without LoRA
            return self.original_layer(x)

    def set_lora_enabled(self, enabled: bool):
        """
        Enable or disable LoRA during inference.

        Args:
            enabled (bool): Flag to enable or disable LoRA.
        """
        self.use_lora = enabled
