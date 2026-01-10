import torch
import torch.nn as nn

class MockUNet(nn.Module):
    """Mimics a small U-Net block."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(8, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # crop or pad to match input length
        diff = x.size(-1) - decoded.size(-1)
        if diff > 0:
            decoded = torch.nn.functional.pad(decoded, (0, diff))
        elif diff < 0:
            decoded = decoded[..., :x.size(-1)]
        return decoded


class Downsampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)



class DownsamplerBN(nn.Module):
    """
    BatchNorm1d after each conv.
    """
    def __init__(self, in_ch=1, mid_ch=8, out_ch=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, mid_ch, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(mid_ch),
            nn.ReLU(),
            nn.Conv1d(mid_ch, out_ch, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)
