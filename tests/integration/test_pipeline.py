"""
End-to-end integration tests for the full preprocessing -> postprocessing pipeline.

These tests verify that:
1. Identity mask approximately reconstructs the original audio
2. Zero mask produces silence
3. Complementary masks sum to approximately the original
4. The full pipeline maintains audio quality (SNR)
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from stemmy.postprocessing.pipeline import Postprocessor
from stemmy.preprocessing.audio import load_audio
from stemmy.preprocessing.pipeline import Preprocessor


@pytest.fixture
def test_audio_path():
    """Path to a test audio file."""
    return Path(__file__).parent.parent / "preprocessor" / "samples" / "plinky_key.wav"


@pytest.fixture
def pipeline_output(test_audio_path):
    """Run preprocessing and return tensor, metadata, and original waveform."""
    prep = Preprocessor()
    tensor, metadata = prep.process(test_audio_path)

    # Load original for comparison (at same sample rate)
    original, sr = load_audio(test_audio_path, sr=metadata.processed_sr)

    # Ensure stereo
    if original.ndim == 1:
        original = np.stack([original, original])

    return tensor, metadata, original


def calculate_snr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Calculate Signal-to-Noise Ratio in dB."""
    # Trim to same length
    min_len = min(original.shape[-1], reconstructed.shape[-1])
    original = original[..., :min_len]
    reconstructed = reconstructed[..., :min_len]

    signal_power = np.sum(original**2)
    noise_power = np.sum((original - reconstructed) ** 2)

    if noise_power == 0:
        return float("inf")

    return 10 * np.log10(signal_power / noise_power)


def test_identity_mask_reconstructs_original(pipeline_output, tmp_path):
    """
    Test that an identity mask (all ones) approximately reconstructs the original.

    This validates the full roundtrip:
    audio -> preprocess -> identity mask -> postprocess -> audio ≈ original
    """
    tensor, metadata, original = pipeline_output

    # Identity mask = all ones (keep everything)
    F, T = tensor.shape[2], tensor.shape[3]
    identity_mask = torch.ones(1, 2, F, T)

    # Run postprocessing
    post = Postprocessor()
    result = post.process(identity_mask, metadata, tensor, tmp_path, "test")

    # Calculate SNR
    snr = calculate_snr(original, result.vocals_waveform)

    # Should have good reconstruction quality
    assert snr > 20, f"SNR too low: {snr:.1f} dB (expected > 20 dB)"


def test_zero_mask_produces_silence(pipeline_output, tmp_path):
    """Test that a zero mask produces near-silent output."""
    tensor, metadata, original = pipeline_output

    # Zero mask = silence everything
    F, T = tensor.shape[2], tensor.shape[3]
    zero_mask = torch.zeros(1, 2, F, T)

    # Run postprocessing
    post = Postprocessor()
    result = post.process(zero_mask, metadata, tensor, tmp_path, "test")

    # Should be near-silent
    energy = np.mean(result.vocals_waveform**2)
    assert energy < 1e-10, f"Zero mask should produce silence, got energy {energy}"


def test_complementary_masks_sum_to_original(pipeline_output, tmp_path):
    """
    Test that complementary masks (vocals + instrumentals) sum to approximately original.

    If vocals_mask + instrumentals_mask = 1 everywhere, then
    vocals_waveform + instrumentals_waveform ≈ original_waveform
    """
    tensor, metadata, original = pipeline_output

    # Create complementary masks (should sum to 1.0)
    F, T = tensor.shape[2], tensor.shape[3]
    vocals_mask = torch.rand(1, 1, F, T) * 0.5 + 0.25  # Random values in [0.25, 0.75]
    instrumentals_mask = 1.0 - vocals_mask

    # Stack into model output format [1, 2, F, T]
    model_output = torch.cat([vocals_mask, instrumentals_mask], dim=1)

    # Run postprocessing
    post = Postprocessor()
    result = post.process(model_output, metadata, tensor, tmp_path, "test")

    # Sum the separated stems
    summed = result.vocals_waveform + result.instrumentals_waveform

    # Should approximately equal original
    snr = calculate_snr(original, summed)
    assert snr > 20, f"Complementary masks sum SNR too low: {snr:.1f} dB"


def test_full_pipeline_creates_valid_files(pipeline_output, tmp_path):
    """Test that the full pipeline creates valid, playable audio files."""
    tensor, metadata, original = pipeline_output

    F, T = tensor.shape[2], tensor.shape[3]
    mock_mask = torch.ones(1, 2, F, T) * 0.5

    post = Postprocessor()
    result = post.process(mock_mask, metadata, tensor, tmp_path, "integration_test")

    # Files should exist
    assert result.vocals_path.exists()
    assert result.instrumentals_path.exists()

    # Files should be readable
    from stemmy.preprocessing.audio import load_audio

    vocals_loaded, sr = load_audio(result.vocals_path, sr=metadata.processed_sr)

    assert sr == metadata.processed_sr
    assert vocals_loaded.shape[-1] > 0


def test_pipeline_preserves_stereo(pipeline_output, tmp_path):
    """Test that stereo information is preserved through the pipeline."""
    tensor, metadata, original = pipeline_output

    F, T = tensor.shape[2], tensor.shape[3]
    identity_mask = torch.ones(1, 2, F, T)

    post = Postprocessor()
    result = post.process(identity_mask, metadata, tensor, tmp_path, "test")

    # Output should be stereo
    assert result.vocals_waveform.shape[0] == 2
    assert result.instrumentals_waveform.shape[0] == 2


def test_pipeline_output_length_matches(pipeline_output, tmp_path):
    """Test that output length matches the original processed length."""
    tensor, metadata, original = pipeline_output

    F, T = tensor.shape[2], tensor.shape[3]
    identity_mask = torch.ones(1, 2, F, T)

    post = Postprocessor()
    result = post.process(identity_mask, metadata, tensor, tmp_path, "test")

    # Output length should match metadata
    assert result.vocals_waveform.shape[1] == metadata.processed_length


def test_pipeline_no_nan_or_inf(pipeline_output, tmp_path):
    """Test that pipeline never produces NaN or Inf values."""
    tensor, metadata, original = pipeline_output

    F, T = tensor.shape[2], tensor.shape[3]

    # Test with various mask values
    test_masks = [
        torch.zeros(1, 2, F, T),
        torch.ones(1, 2, F, T),
        torch.rand(1, 2, F, T),
    ]

    post = Postprocessor()

    for mask in test_masks:
        result = post.process(mask, metadata, tensor, tmp_path, "test", export_files=False)

        assert not np.any(np.isnan(result.vocals_waveform)), "NaN in vocals"
        assert not np.any(np.isinf(result.vocals_waveform)), "Inf in vocals"
        assert not np.any(np.isnan(result.instrumentals_waveform)), "NaN in instrumentals"
        assert not np.any(np.isinf(result.instrumentals_waveform)), "Inf in instrumentals"
