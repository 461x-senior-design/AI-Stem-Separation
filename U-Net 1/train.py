"""
Training Pipeline for U-Net Audio Source Separation (MUSDB18-style)

Complete training pipeline that:
1. Loads MUSDB18-style dataset (mixture + 4 stems: drums, bass, vocals, other)
2. Trains U-Net to predict 8-channel output (4 stems x 2 stereo)
3. Saves model weights for inference

Dataset structure expected:
    dataset/
    |-- train/
    |   |-- song1/
    |   |   |-- mixture.wav
    |   |   |-- drums.wav
    |   |   |-- bass.wav
    |   |   |-- vocals.wav
    |   |   |-- other.wav
    |   |-- song2/
    |   |   |-- ...
    |-- test/
    |   |-- ...

Usage:
    python train.py --data-dir /path/to/dataset --epochs 100

Author: Q1 U-Net Implementation
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models import UNet
from data.preprocessing import AudioPreprocessor
from utils.losses import get_loss_function
from utils.config import get_config, get_device

# Check for audio dependencies
try:
    import librosa
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("WARNING: librosa/soundfile not installed. Audio loading disabled.")


# =============================================================================
# CONSTANTS
# =============================================================================

STEM_NAMES = ['drums', 'bass', 'vocals', 'other']
NUM_STEMS = len(STEM_NAMES)
STEREO_CHANNELS = 2
OUTPUT_CHANNELS = NUM_STEMS * STEREO_CHANNELS  # 8 channels total


# =============================================================================
# DATASET
# =============================================================================

class MUSDB18Dataset(Dataset):
    """
    Dataset for MUSDB18-style audio source separation.

    Loads mixture and stem audio files, converts to spectrograms.

    Args:
        data_dir: Path to dataset directory (e.g., 'dataset/train')
        preprocessor: AudioPreprocessor instance for STFT conversion
        segment_length: Number of time frames per segment (for chunking long songs)
        overlap: Overlap between segments (0.0 to 0.5)
        augment: Whether to apply data augmentation

    Each sample returns:
        mixture: (2, freq, time) - stereo mixture spectrogram
        stems: (8, freq, time) - all stems concatenated (drums, bass, vocals, other x stereo)
        metadata: dict with song info
    """

    def __init__(
        self,
        data_dir: str,
        preprocessor: AudioPreprocessor,
        segment_length: int = 256,
        overlap: float = 0.25,
        augment: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.preprocessor = preprocessor
        self.segment_length = segment_length
        self.overlap = overlap
        self.augment = augment

        # Find all songs (directories with mixture.wav)
        self.songs = self._discover_songs()

        # Build segment index for efficient random access
        self.segments = self._build_segment_index()

        print(f"MUSDB18Dataset: Found {len(self.songs)} songs, {len(self.segments)} segments")

    def _discover_songs(self) -> List[Path]:
        """Find all valid song directories."""
        songs = []

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        for song_dir in sorted(self.data_dir.iterdir()):
            if not song_dir.is_dir():
                continue

            # Check for required files
            mixture_path = song_dir / 'mixture.wav'
            if not mixture_path.exists():
                continue

            # Check for all stems
            stems_present = all(
                (song_dir / f'{stem}.wav').exists()
                for stem in STEM_NAMES
            )

            if stems_present:
                songs.append(song_dir)
            else:
                print(f"WARNING: Skipping {song_dir.name} - missing stems")

        return songs

    def _build_segment_index(self) -> List[Tuple[int, int]]:
        """
        Build index of (song_idx, segment_idx) for all segments.

        Precomputes segment boundaries for efficient access.
        """
        segments = []
        stride = int(self.segment_length * (1 - self.overlap))

        for song_idx, song_dir in enumerate(self.songs):
            # Load mixture to get duration
            mixture_path = song_dir / 'mixture.wav'

            if AUDIO_AVAILABLE:
                # Get duration without loading full audio
                info = sf.info(str(mixture_path))
                duration_samples = int(info.samplerate * info.duration)

                # Estimate number of time frames after STFT
                n_frames = 1 + (duration_samples - self.preprocessor.n_fft) // self.preprocessor.hop_length

                # Calculate segments
                n_segments = max(1, (n_frames - self.segment_length) // stride + 1)

                for seg_idx in range(n_segments):
                    segments.append((song_idx, seg_idx))
            else:
                # Fallback: single segment per song
                segments.append((song_idx, 0))

        return segments

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Get a training sample.

        Returns:
            mixture: (2, freq, segment_length) mixture spectrogram
            stems: (8, freq, segment_length) concatenated stem spectrograms
            metadata: dict with song name, segment info
        """
        song_idx, seg_idx = self.segments[idx]
        song_dir = self.songs[song_idx]

        stride = int(self.segment_length * (1 - self.overlap))
        start_frame = seg_idx * stride
        end_frame = start_frame + self.segment_length

        # Load and preprocess mixture
        mixture_path = song_dir / 'mixture.wav'
        mixture_tensor, mixture_phase, mixture_meta = self.preprocessor.preprocess(str(mixture_path))
        mixture = mixture_tensor.squeeze(0)  # Remove batch dim: (2, freq, time)

        # Load and preprocess each stem
        stem_list = []
        for stem_name in STEM_NAMES:
            stem_path = song_dir / f'{stem_name}.wav'
            stem_tensor, _, _ = self.preprocessor.preprocess(str(stem_path))
            stem_list.append(stem_tensor.squeeze(0))  # (2, freq, time)

        # Concatenate stems: (8, freq, time)
        stems = torch.cat(stem_list, dim=0)

        # Extract segment
        freq_bins = mixture.shape[1]
        time_frames = mixture.shape[2]

        # Handle case where segment exceeds audio length
        if end_frame > time_frames:
            # Pad with zeros if needed
            pad_amount = end_frame - time_frames
            mixture = torch.nn.functional.pad(mixture, (0, pad_amount))
            stems = torch.nn.functional.pad(stems, (0, pad_amount))

        # Extract segment
        mixture = mixture[:, :, start_frame:end_frame]
        stems = stems[:, :, start_frame:end_frame]

        # Data augmentation
        if self.augment:
            mixture, stems = self._augment(mixture, stems)

        metadata = {
            'song_name': song_dir.name,
            'song_idx': song_idx,
            'segment_idx': seg_idx,
            'start_frame': start_frame,
            'end_frame': end_frame
        }

        return mixture, stems, metadata

    def _augment(
        self,
        mixture: torch.Tensor,
        stems: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply data augmentation.

        Currently supports:
        - Random gain scaling per stem
        - Random left/right swap
        """
        # Random gain per stem (0.5 to 1.5)
        if torch.rand(1).item() < 0.5:
            new_stems = []
            new_mixture = torch.zeros_like(mixture)

            for i in range(NUM_STEMS):
                gain = 0.5 + torch.rand(1).item()
                stem = stems[i*2:(i+1)*2] * gain
                new_stems.append(stem)
                new_mixture += stem

            stems = torch.cat(new_stems, dim=0)
            mixture = new_mixture

        # Random left/right swap
        if torch.rand(1).item() < 0.3:
            # Swap stereo channels
            mixture = mixture.flip(0)
            stems_list = []
            for i in range(NUM_STEMS):
                stem = stems[i*2:(i+1)*2].flip(0)
                stems_list.append(stem)
            stems = torch.cat(stems_list, dim=0)

        return mixture, stems


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss, 'max' for metrics like accuracy
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation metric

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False


class TrainingLogger:
    """
    Logging utility for training progress.

    Saves training history to JSON and prints progress.
    """

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_time': []
        }

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        lr: float,
        epoch_time: float
    ) -> None:
        """Log epoch metrics."""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['learning_rate'].append(lr)
        self.history['epoch_time'].append(epoch_time)

        print(f"Epoch {epoch:4d} | "
              f"Train: {train_loss:.6f} | "
              f"Val: {val_loss:.6f} | "
              f"LR: {lr:.2e} | "
              f"Time: {epoch_time:.1f}s")

    def save(self, filename: str = 'training_history.json') -> None:
        """Save training history to JSON."""
        path = self.log_dir / filename
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {path}")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    val_loss: float,
    config: Dict[str, Any],
    checkpoint_dir: str,
    is_best: bool = False
) -> str:
    """
    Save model checkpoint.

    Args:
        model: The U-Net model
        optimizer: Optimizer state
        scheduler: Learning rate scheduler state
        epoch: Current epoch
        val_loss: Validation loss
        config: Training configuration
        checkpoint_dir: Directory to save checkpoints
        is_best: If True, also save as 'best_model.pth'

    Returns:
        Path to saved checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_loss': val_loss,
        'config': config
    }

    # Save periodic checkpoint
    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pth'
    torch.save(checkpoint, checkpoint_path)

    # Save best model
    if is_best:
        best_path = checkpoint_dir / 'best_model.pth'
        torch.save(checkpoint, best_path)
        print(f"New best model saved: val_loss = {val_loss:.6f}")

    # Save latest (for resume)
    latest_path = checkpoint_dir / 'latest.pth'
    torch.save(checkpoint, latest_path)

    return str(checkpoint_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: torch.device = None
) -> Dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to restore
        scheduler: Optional scheduler to restore
        device: Device to load onto

    Returns:
        Checkpoint dictionary with epoch, val_loss, config
    """
    if device is None:
        device = get_device()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, val_loss = {checkpoint['val_loss']:.6f}")

    return checkpoint


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0
) -> float:
    """
    Train for one epoch.

    Args:
        model: U-Net model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        grad_clip: Gradient clipping value

    Returns:
        Average training loss for epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (mixture, stems, _) in enumerate(train_loader):
        mixture = mixture.to(device)
        stems = stems.to(device)

        optimizer.zero_grad()

        # Forward pass
        predictions = model(mixture)

        # Compute loss
        loss = criterion(predictions, stems)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Progress indicator
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}", end='\r')

    return total_loss / max(num_batches, 1)


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Validate for one epoch.

    Args:
        model: U-Net model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on

    Returns:
        Average validation loss for epoch
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for mixture, stems, _ in val_loader:
            mixture = mixture.to(device)
            stems = stems.to(device)

            predictions = model(mixture)
            loss = criterion(predictions, stems)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    checkpoint_dir: str,
    log_dir: str,
    resume_from: Optional[str] = None
) -> nn.Module:
    """
    Full training loop.

    Args:
        model: U-Net model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary
        device: Device to train on
        checkpoint_dir: Directory for checkpoints
        log_dir: Directory for logs
        resume_from: Optional checkpoint path to resume from

    Returns:
        Trained model
    """
    # Setup loss function
    criterion = get_loss_function(
        config['loss']['type'],
        alpha=config['loss']['combined_alpha'],
        beta=config['loss']['combined_beta']
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Setup scheduler
    scheduler_type = config['training'].get('scheduler', 'reduce_on_plateau')
    if scheduler_type == 'reduce_on_plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['training']['scheduler_factor'],
            patience=config['training']['scheduler_patience'],
            verbose=True
        )
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs']
        )
    else:
        scheduler = None

    # Setup logging and early stopping
    logger = TrainingLogger(log_dir)
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping_patience'],
        mode='min'
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')

    if resume_from:
        checkpoint = load_checkpoint(resume_from, model, optimizer, scheduler, device)
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']

    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Loss: {config['loss']['type']}")
    print("=" * 70 + "\n")

    for epoch in range(start_epoch, config['training']['epochs']):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip=config['training'].get('grad_clip', 1.0)
        )

        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Update scheduler
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Log epoch
        epoch_time = time.time() - epoch_start
        logger.log_epoch(epoch, train_loss, val_loss, current_lr, epoch_time)

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        save_checkpoint(
            model, optimizer, scheduler, epoch, val_loss,
            config, checkpoint_dir, is_best=is_best
        )

        # Early stopping check
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    # Save final training history
    logger.save()

    # Load best model
    best_path = Path(checkpoint_dir) / 'best_model.pth'
    if best_path.exists():
        load_checkpoint(str(best_path), model, device=device)

    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {checkpoint_dir}")
    print("=" * 70)

    return model


# =============================================================================
# INFERENCE
# =============================================================================

def separate_stems(
    model: nn.Module,
    audio_path: str,
    output_dir: str,
    preprocessor: AudioPreprocessor,
    device: torch.device,
    overlap: float = 0.5
) -> Dict[str, str]:
    """
    Separate a song into its constituent stems.

    Args:
        model: Trained U-Net model
        audio_path: Path to input mixture audio file
        output_dir: Directory to save separated stems
        preprocessor: AudioPreprocessor instance
        device: Device to run inference on
        overlap: Overlap between windows for smoother output

    Returns:
        Dictionary mapping stem names to output file paths
    """
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_path = Path(audio_path)
    song_name = audio_path.stem

    print(f"Separating: {audio_path.name}")

    # Preprocess mixture
    mixture_tensor, mixture_phase, metadata = preprocessor.preprocess(str(audio_path))
    mixture_tensor = mixture_tensor.to(device)  # (1, 2, freq, time)

    freq_bins, time_frames = mixture_tensor.shape[2], mixture_tensor.shape[3]
    segment_length = 256  # Match training segment length

    # Process in overlapping windows
    stride = int(segment_length * (1 - overlap))
    output_stems = torch.zeros(1, OUTPUT_CHANNELS, freq_bins, time_frames, device=device)
    weight_sum = torch.zeros(1, 1, 1, time_frames, device=device)

    # Create smooth window for blending
    window = torch.hann_window(segment_length, device=device)
    window = window.view(1, 1, 1, -1)

    with torch.no_grad():
        for start in range(0, time_frames, stride):
            end = min(start + segment_length, time_frames)
            actual_len = end - start

            # Extract segment
            segment = mixture_tensor[:, :, :, start:end]

            # Pad if needed
            if actual_len < segment_length:
                pad_amount = segment_length - actual_len
                segment = torch.nn.functional.pad(segment, (0, pad_amount))

            # Forward pass
            pred = model(segment)  # (1, 8, freq, segment_length)

            # Unpad if we padded
            if actual_len < segment_length:
                pred = pred[:, :, :, :actual_len]
                current_window = window[:, :, :, :actual_len]
            else:
                current_window = window

            # Accumulate with window weighting
            output_stems[:, :, :, start:end] += pred * current_window
            weight_sum[:, :, :, start:end] += current_window

    # Normalize by window sum
    output_stems = output_stems / (weight_sum + 1e-8)

    # Convert to numpy and reshape
    output_stems = output_stems.cpu().numpy()[0]  # (8, freq, time)

    # Save each stem
    output_paths = {}
    for i, stem_name in enumerate(STEM_NAMES):
        # Extract stereo channels for this stem
        stem_magnitude = output_stems[i*2:(i+1)*2]  # (2, freq, time)

        # Reconstruct audio using mixture phase
        stem_audio = preprocessor.reconstruct_audio(
            stem_magnitude,
            mixture_phase,
            metadata['norm_params']
        )

        # Save to file
        output_path = output_dir / f'{song_name}_{stem_name}.wav'
        sf.write(str(output_path), stem_audio.T, preprocessor.sr)
        output_paths[stem_name] = str(output_path)
        print(f"  Saved: {output_path.name}")

    print(f"Separation complete! Files saved to: {output_dir}")
    return output_paths


def load_model_for_inference(
    checkpoint_path: str,
    device: torch.device = None
) -> Tuple[nn.Module, AudioPreprocessor, Dict[str, Any]]:
    """
    Load a trained model for inference.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load onto (defaults to best available)

    Returns:
        Tuple of (model, preprocessor, config)
    """
    if device is None:
        device = get_device()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', get_config())

    # Create model with saved config
    model = UNet(
        input_channels=config['model']['input_channels'],
        output_channels=OUTPUT_CHANNELS,  # 8 for 4 stems
        initial_filters=config['model']['initial_filters'],
        depth=config['model']['depth'],
        use_dropout=config['model']['use_dropout'],
        dropout_prob=config['model']['dropout_prob']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create preprocessor
    preprocessor = AudioPreprocessor(
        n_fft=config['audio']['n_fft'],
        hop_length=config['audio']['hop_length'],
        sr=config['audio']['sample_rate'],
        normalize=config['audio']['normalize']
    )

    print(f"Loaded model from epoch {checkpoint['epoch']}, val_loss = {checkpoint['val_loss']:.6f}")

    return model, preprocessor, config


# =============================================================================
# MAIN
# =============================================================================

def collate_fn(batch):
    """
    Custom collate function that handles metadata.

    Stacks tensors and groups metadata into a list.
    """
    mixtures = torch.stack([item[0] for item in batch])
    stems = torch.stack([item[1] for item in batch])
    metadata = [item[2] for item in batch]
    return mixtures, stems, metadata


def main():
    parser = argparse.ArgumentParser(
        description='Train U-Net for audio source separation'
    )
    parser.add_argument(
        '--data-dir', type=str, required=True,
        help='Path to MUSDB18-style dataset directory'
    )
    parser.add_argument(
        '--output-dir', type=str, default='./output',
        help='Directory for outputs (checkpoints, logs)'
    )
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Number of training epochs (overrides config)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--lr', type=float, default=None,
        help='Learning rate (overrides config)'
    )
    parser.add_argument(
        '--loss', type=str, default=None, choices=['l1', 'mse', 'combined'],
        help='Loss function (overrides config)'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--segment-length', type=int, default=256,
        help='Time frames per training segment'
    )
    parser.add_argument(
        '--val-split', type=float, default=0.1,
        help='Fraction of data to use for validation'
    )
    parser.add_argument(
        '--no-augment', action='store_true',
        help='Disable data augmentation'
    )

    args = parser.parse_args()

    # Load configuration
    config = get_config()

    # Override config with command line arguments
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.loss:
        config['loss']['type'] = args.loss

    # Update output channels for 4 stems
    config['model']['output_channels'] = OUTPUT_CHANNELS

    # Setup paths
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / 'checkpoints'
    log_dir = output_dir / 'logs'

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Create preprocessor
    preprocessor = AudioPreprocessor(
        n_fft=config['audio']['n_fft'],
        hop_length=config['audio']['hop_length'],
        sr=config['audio']['sample_rate'],
        normalize=config['audio']['normalize']
    )

    # Create dataset
    data_path = Path(args.data_dir)
    train_data_path = data_path / 'train'

    if not train_data_path.exists():
        # Try using the data_dir directly if no train subdirectory
        train_data_path = data_path
        print(f"Note: Using {data_path} directly (no 'train' subdirectory found)")

    full_dataset = MUSDB18Dataset(
        data_dir=str(train_data_path),
        preprocessor=preprocessor,
        segment_length=args.segment_length,
        overlap=0.25,
        augment=not args.no_augment
    )

    # Split into train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        collate_fn=collate_fn
    )

    # Create model
    model = UNet(
        input_channels=config['model']['input_channels'],
        output_channels=OUTPUT_CHANNELS,
        initial_filters=config['model']['initial_filters'],
        depth=config['model']['depth'],
        use_dropout=config['model']['use_dropout'],
        dropout_prob=config['model']['dropout_prob']
    ).to(device)

    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Model size: {model.get_model_size_mb():.2f} MB")

    # Train
    trained_model = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_dir=str(checkpoint_dir),
        log_dir=str(log_dir),
        resume_from=args.resume
    )

    print("\nTraining complete!")
    print(f"Best model saved to: {checkpoint_dir / 'best_model.pth'}")
    print("\nTo separate a song:")
    print(f"  python train.py --separate /path/to/song.wav --checkpoint {checkpoint_dir / 'best_model.pth'}")


def separate_main():
    """Entry point for separation mode."""
    parser = argparse.ArgumentParser(
        description='Separate audio into stems using trained model'
    )
    parser.add_argument(
        '--separate', type=str, required=True,
        help='Path to audio file to separate'
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--output-dir', type=str, default='./separated',
        help='Directory to save separated stems'
    )
    parser.add_argument(
        '--overlap', type=float, default=0.5,
        help='Overlap between processing windows (0.0 to 0.9)'
    )

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Load model
    model, preprocessor, config = load_model_for_inference(
        args.checkpoint, device
    )

    # Separate
    output_paths = separate_stems(
        model=model,
        audio_path=args.separate,
        output_dir=args.output_dir,
        preprocessor=preprocessor,
        device=device,
        overlap=args.overlap
    )

    print("\nSeparated stems:")
    for stem_name, path in output_paths.items():
        print(f"  {stem_name}: {path}")


if __name__ == '__main__':
    # Check if in separation mode
    if '--separate' in sys.argv:
        separate_main()
    else:
        main()
