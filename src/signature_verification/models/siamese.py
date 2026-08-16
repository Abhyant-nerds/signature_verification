from __future__ import annotations

import timm
import torch
from torch import nn
from torch.nn import functional as F


class SiameseNetwork(nn.Module):
    def __init__(
        self,
        encoder_name: str = "convnext_tiny",
        embedding_dim: int = 256,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.encoder_name = encoder_name
        self.embedding_dim = embedding_dim
        self.backbone = timm.create_model(
            encoder_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        feature_dim = self.backbone.num_features
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(self.backbone(image)), p=2, dim=1)

    def forward(self, left: torch.Tensor, right: torch.Tensor):
        return self.encode(left), self.encode(right)
