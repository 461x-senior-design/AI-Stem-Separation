import torch
import torch.nn as nn


class MockEncoder(nn.Module):
    """
    Simple mock encoder to simulate U-Net downsampling.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(8, 16, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )

    def forward(self, x):
        return self.net(x)
