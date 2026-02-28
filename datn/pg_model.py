"""
Prompt Generator (PG): predict per-slice objectness + bbox from 2.5D context.

Architecture:  ResNet18 (conv1 → 9 channels) + two heads:
  - objectness head: Global Average Pool → FC → 1 logit
  - bbox head:       Global Average Pool → FC → 4 values (normalised x1,y1,x2,y2)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class PromptGenerator(nn.Module):
    """
    Input:  (B, 9, H, W) — 3 modalities × 3 consecutive slices
    Output: objectness (B, 1) logit, bbox (B, 4) normalised [0,1]
    """

    def __init__(self, in_channels: int = 9, pretrained_backbone: bool = True):
        super().__init__()

        # --- backbone: ResNet18 with modified first conv ---
        weights = models.ResNet18_Weights.DEFAULT if pretrained_backbone else None
        backbone = models.resnet18(weights=weights)

        # Replace conv1: 3ch → in_channels
        old_conv = backbone.conv1
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Initialise: replicate pretrained weights across extra channels
        if pretrained_backbone:
            with torch.no_grad():
                # repeat 3ch weights to fill 9ch (3×3)
                rep = in_channels // 3
                remainder = in_channels % 3
                parts = [old_conv.weight.data] * rep
                if remainder:
                    parts.append(old_conv.weight.data[:, :remainder])
                self.conv1.weight.copy_(torch.cat(parts, dim=1))

        self.bn1   = backbone.bn1
        self.relu  = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        feat_dim = 512  # ResNet18 last layer

        # --- heads ---
        self.obj_head = nn.Linear(feat_dim, 1)
        self.bbox_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.Sigmoid(),  # output in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        feat = x.flatten(1)  # (B, 512)

        obj_logit = self.obj_head(feat)          # (B, 1)
        bbox      = self.bbox_head(feat)         # (B, 4)

        return {"objectness": obj_logit.squeeze(-1), "bbox": bbox}
