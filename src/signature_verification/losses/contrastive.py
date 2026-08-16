from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ContrastiveLoss(nn.Module):
    """Cosine-distance contrastive loss; label 1 means a matching pair."""

    def __init__(self, margin: float = 0.5) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self, left_embedding: torch.Tensor, right_embedding: torch.Tensor, label: torch.Tensor
    ) -> torch.Tensor:
        distance = 1.0 - F.cosine_similarity(left_embedding, right_embedding)
        positive = label * distance.pow(2)
        negative = (1.0 - label) * F.relu(self.margin - distance).pow(2)
        return 0.5 * (positive + negative).mean()
