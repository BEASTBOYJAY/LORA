import torch

import torch.nn as nn


class LoRA(nn.Module):
    def __init__(self, original_layer, alpha: float = 0.5, rank: int = 4):
        super(LoRA, self).__init__()
        self.original_layer = original_layer
        self.rank = rank

        self.lora_A = nn.Parameter(
            torch.randn(rank, original_layer.in_features) * alpha
        )
        self.lora_B = nn.Parameter(
            torch.randn(original_layer.out_features, rank) * alpha
        )

        self.use_lora = True

    def forward(self, x):
        if self.use_lora:
            # Apply the original layer + low-rank adaptation
            return self.original_layer(x) + torch.matmul(
                torch.matmul(x, self.lora_A.T), self.lora_B.T
            )
        else:
            # Only apply the original layer withouto LoRA
            return self.original_layer(x)

    def set_lora_enabled(self, enabled: bool):
        """Enable or disable LoRA during inference."""
        self.use_lora = enabled
