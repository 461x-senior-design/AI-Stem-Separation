"""
Minimal U-Net Training Script for Vocal Separation
Uses only PyTorch and torchaudio - no librosa
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import torch.nn.functional as F


class UNet(nn.Module):
    """U-Net with skip connections"""

    def __init__(self):
        super(UNet, self).__init__()

        self.enc1 = self.conv_block(1, 16)
        self.enc2 = self.conv_block(16, 32)
        self.enc3 = self.conv_block(32, 64)
        self.enc4 = self.conv_block(64, 128)

        self.bottleneck = self.conv_block(128, 256)

        self.upconv4 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec4 = self.conv_block(256, 128)

        self.upconv3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec3 = self.conv_block(128, 64)

        self.upconv2 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec2 = self.conv_block(64, 32)

        self.upconv1 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.dec1 = self.conv_block(32, 16)

        self.out = nn.Conv2d(16, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        return self.out(dec1)


class VocalDataset(Dataset):
    """
    Dataset structure:
    dataset_root/
        train/
            mix/*.wav
            vocals/*.wav
        val/
            mix/*.wav
            vocals/*.wav
    """

    def __init__(self, root, split="train", chunk_size=22050 * 5):
        self.mix_dir = Path(root) / split / "mix"
        self.vocals_dir = Path(root) / split / "vocals"
        self.chunk_size = chunk_size
        self.files = sorted([f.stem for f in self.mix_dir.glob("*.wav")])

        if not self.files:
            raise ValueError(f"No files in {self.mix_dir}")

        print(f"{split}: {len(self.files)} files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        # Load audio
        # Load with soundfile (returns np arrays: shape [num_frames, num_channels])
        mix_np, sr = sf.read(self.mix_dir / f"{name}.wav", always_2d=True)
        voc_np, _ = sf.read(self.vocals_dir / f"{name}.wav", always_2d=True)

        # -> torch, shape [channels, num_frames], float32
        mix = torch.from_numpy(mix_np).T.contiguous().to(torch.float32)
        vocals = torch.from_numpy(voc_np).T.contiguous().to(torch.float32)

        # convert to mono
        if mix.shape[0] > 1:
            mix = mix.mean(0, keepdim=True)
        if vocals.shape[0] > 1:
            vocals = vocals.mean(0, keepdim=True)

        # random crop / pad (unchanged)
        if mix.shape[1] > self.chunk_size:
            start = torch.randint(0, mix.shape[1] - self.chunk_size, (1,)).item()
            mix = mix[:, start : start + self.chunk_size]
            vocals = vocals[:, start : start + self.chunk_size]
        elif mix.shape[1] < self.chunk_size:
            pad = self.chunk_size - mix.shape[1]
            mix = F.pad(mix, (0, pad))
            vocals = F.pad(vocals, (0, pad))

        # STFT
        win = torch.hann_window(2048, device=mix.device, dtype=mix.dtype)
        mix_spec = torch.stft(
            mix,
            n_fft=2048,
            hop_length=512,
            win_length=2048,
            window=win,
            center=True,
            return_complex=True,
            normalized=True,
        )

        win_v = torch.hann_window(2048, device=vocals.device, dtype=vocals.dtype)
        vocals_spec = torch.stft(
            vocals,
            n_fft=2048,
            hop_length=512,
            win_length=2048,
            window=win_v,
            center=True,
            return_complex=True,
            normalized=True,
        )

        # Magnitude
        mix_mag = torch.abs(mix_spec)
        vocals_mag = torch.abs(vocals_spec)

        # Normalize
        mix_mag = (mix_mag - mix_mag.mean()) / (mix_mag.std() + 1e-8)
        vocals_mag = (vocals_mag - vocals_mag.mean()) / (vocals_mag.std() + 1e-8)

        return mix_mag, vocals_mag


def collate_fn(batch):
    """Pad to same size and ensure divisible by 16"""
    mixes, vocals = zip(*batch)

    max_h = max(m.shape[1] for m in mixes)
    max_w = max(m.shape[2] for m in mixes)

    # Pad to multiple of 16
    pad_h = (16 - max_h % 16) % 16
    pad_w = (16 - max_w % 16) % 16
    max_h += pad_h
    max_w += pad_w

    mixes_out = []
    vocals_out = []

    for mix, vocal in zip(mixes, vocals):
        h_pad = max_h - mix.shape[1]
        w_pad = max_w - mix.shape[2]

        mix_padded = nn.functional.pad(mix, (0, w_pad, 0, h_pad))
        vocal_padded = nn.functional.pad(vocal, (0, w_pad, 0, h_pad))

        mixes_out.append(mix_padded)
        vocals_out.append(vocal_padded)

    return torch.stack(mixes_out), torch.stack(vocals_out)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for mix, vocals in tqdm(loader, desc="Train"):
        mix, vocals = mix.to(device), vocals.to(device)

        optimizer.zero_grad()
        pred = model(mix)
        loss = criterion(pred, vocals)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for mix, vocals in tqdm(loader, desc="Val"):
            mix, vocals = mix.to(device), vocals.to(device)
            pred = model(mix)
            loss = criterion(pred, vocals)
            total_loss += loss.item()

    return total_loss / len(loader)


def train(dataset_root, epochs=50, batch_size=8, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Data
    train_data = VocalDataset(dataset_root, "train")
    val_data = VocalDataset(dataset_root, "val")

    train_loader = DataLoader(
        train_data, batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_data, batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    # Model
    model = UNet().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    # Training
    best_loss = float("inf")
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        # Save
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            "checkpoints/latest.pth",
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"model": model.state_dict()}, "checkpoints/best.pth")
            print(f"✓ Best model saved")

    print(f"\nDone! Best val loss: {best_loss:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset root")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()
    train(args.dataset, args.epochs, args.batch_size, args.lr)
