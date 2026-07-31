from __future__ import annotations

import torch
import torch.nn as nn


FOOD_CLASSES = (
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles/Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetables/Fruit",
)


class CNNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FoodClassifier(nn.Module):
    """The same residual CNN used by this workspace's HW3 solution."""

    def __init__(self, num_classes: int = len(FOOD_CLASSES)) -> None:
        super().__init__()
        self.cnn_layer1 = CNNBlock(3, 64, 3, 1, 1)
        self.cnn_layer2 = CNNBlock(64, 64, 3, 1, 1)
        self.cnn_layer3 = CNNBlock(64, 128, 3, 2, 1)
        self.cnn_layer4 = CNNBlock(128, 128, 3, 1, 1)
        self.cnn_layer5 = CNNBlock(128, 256, 3, 2, 1)
        self.cnn_layer6 = CNNBlock(256, 256, 3, 1, 1)

        self.shortcut2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128),
        )
        self.shortcut3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.cnn_layer1(x))
        residual = x
        x = self.cnn_layer2(x)
        x = self.relu(x + residual)

        residual = self.shortcut2(x)
        x = self.relu(self.cnn_layer3(x))
        x = self.cnn_layer4(x)
        x = self.relu(x + residual)

        residual = self.shortcut3(x)
        x = self.relu(self.cnn_layer5(x))
        x = self.cnn_layer6(x)
        x = self.relu(x + residual)
        return self.fc_layer(self.pool(x))


def find_conv_layer(model: nn.Module, name: str) -> nn.Conv2d:
    modules = dict(model.named_modules())
    if name not in modules:
        available = [key for key, value in modules.items() if isinstance(value, nn.Conv2d)]
        raise ValueError(f"Unknown convolution layer '{name}'. Available: {available}")
    layer = modules[name]
    if not isinstance(layer, nn.Conv2d):
        raise ValueError(f"'{name}' is not a Conv2d layer")
    return layer
