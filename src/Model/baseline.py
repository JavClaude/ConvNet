from typing import Tuple

import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"


class ConvNet(nn.Module):
    """Baseline Convolutional Neural Network for MNIST dataset

    Args:
        channels_in (int): number of input's channels
        n_classes (int) number of classes to predict
    """
    def __init__(self,
                channels_in: int,
                n_classes: int) -> None:
        super(ConvNet, self).__init__()

        # [(W−K+2P)/S]+1

        self.first_conv_block = nn.Sequential(
            nn.Conv2d(in_channels=channels_in, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
        )

        self.first_pooling_block = nn.AvgPool2d(kernel_size=(2, 2))

        self.second_conv_block = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
        )

        self.second_pooling_block = nn.MaxPool2d(kernel_size=(2, 2))

        self.third_conv_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
        )

        self.linear_block = nn.Sequential(
            nn.Linear(6272, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, inputs: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        inputs = self.first_conv_block(inputs)
        inputs = self.first_pooling_block(inputs)

        inputs = self.second_conv_block(inputs)
        inputs = self.second_pooling_block(inputs)

        inputs = self.third_conv_block(inputs)
        pre_logits = self.linear_block(inputs.view(inputs.size()[0], -1))

        return inputs, pre_logits
