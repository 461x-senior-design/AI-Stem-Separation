from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from stemmy.constants import TARGET_CHANNELS
from stemmy.postprocessing.pipeline import Postprocessor, SeparationResult
from stemmy.preprocessing.pipeline import Preprocessor


@pytest.fixture
def test_audio_path():
    """Path to a test audio file."""
    return Path(__file__).parent.parent / "preprocessor" / "samples" / "plinky_key.wav"


@pytest.fixture
def preprocessed_data(test_audio_path):
    """Preprocess a test file and return tensor + metadata."""
    prep = Preprocessor()
    tensor, metadata = prep.process(test_audio_path)
    return tensor, metadata


def metadata_without_mix_magnitude(metadata):
    """Create reconstruction metadata without stored mix magnitude."""
    values = {
        "spectrogram_norm_params": {"method": "none"},
        "phase": metadata.phase,
        "hop_length": metadata.hop_length,
        "processed_length": metadata.processed_length,
        "processed_sr": metadata.processed_sr,
        "n_fft": metadata.n_fft,
        "waveform_norm_params": metadata.waveform_norm_params,
    }

    if hasattr(metadata, "win_length"):
        values["win_length"] = metadata.win_length

    if hasattr(metadata, "center"):
        values["center"] = metadata.center

    return SimpleNamespace(**values)


def test_postprocessor_output_type(preprocessed_data, tmp_path):
    """Test that process() returns SeparationResult."""
    # Arrange
    tensor, metadata = preprocessed_data
    post = Postprocessor()

    # Create mock model output (identity mask = all ones)
    F, T = tensor.shape[2], tensor.shape[3]
    mock_output = torch.ones(1, 2, F, T)

    # Act
    result = post.process(mock_output, metadata, tensor, tmp_path, "test")

    # Assert
    assert isinstance(result, SeparationResult)


def test_postprocessor_creates_files(preprocessed_data, tmp_path):
    """Test that process() creates vocal and instrumental files."""
    # Arrange
    tensor, metadata = preprocessed_data
    post = Postprocessor()

    F, T = tensor.shape[2], tensor.shape[3]
    mock_output = torch.ones(1, 2, F, T) * 0.5

    # Act
    result = post.process(mock_output, metadata, tensor, tmp_path, "test")

    # Assert
    assert result.vocals_path is not None
    assert result.instrumentals_path is not None
    assert result.vocals_path.exists()
    assert result.instrumentals_path.exists()


def test_postprocessor_file_naming(preprocessed_data, tmp_path):
    """Test that output files are named correctly."""
    # Arrange
    tensor, metadata = preprocessed_data
    post = Postprocessor()

    F, T = tensor.shape[2], tensor.shape[3]
    mock_output = torch.ones(1, 2, F, T) * 0.5

    # Act
    result = post.process(mock_output, metadata, tensor, tmp_path, "mysong")

    # Assert
    assert result.vocals_path.name == "mysong_vocals.wav"
    assert result.instrumentals_path.name == "mysong_instrumentals.wav"


def test_postprocessor_waveform_shape(preprocessed_data, tmp_path):
    """Test that output waveforms have correct shape."""
    # Arrange
    tensor, metadata = preprocessed_data
    post = Postprocessor()

    F, T = tensor.shape[2], tensor.shape[3]
    mock_output = torch.ones(1, 2, F, T) * 0.5

    # Act
    result = post.process(mock_output, metadata, tensor, tmp_path, "test")

    # Assert - should be stereo [2, N]
    assert result.vocals_waveform.ndim == 2
    assert result.vocals_waveform.shape[0] == 2
    assert result.instrumentals_waveform.ndim == 2
    assert result.instrumentals_waveform.shape[0] == 2


def test_postprocessor_waveform_length(preprocessed_data, tmp_path):
    """Test that output waveforms match original length."""
    # Arrange
    tensor, metadata = preprocessed_data
    post = Postprocessor()

    F, T = tensor.shape[2], tensor.shape[3]
    mock_output = torch.ones(1, 2, F, T) * 0.5

    # Act
    result = post.process(mock_output, metadata, tensor, tmp_path, "test")

    # Assert - should match processed_length from metadata
    assert result.vocals_waveform.shape[1] == metadata.processed_length
    assert result.instrumentals_waveform.shape[1] == metadata.processed_length


def test_postprocessor_sample_rate(preprocessed_data, tmp_path):
    """Test that result has correct sample rate."""
    # Arrange
    tensor, metadata = preprocessed_data
    post = Postprocessor()

    F, T = tensor.shape[2], tensor.shape[3]
    mock_output = torch.ones(1, 2, F, T) * 0.5

    # Act
    result = post.process(mock_output, metadata, tensor, tmp_path, "test")

    # Assert
    assert result.sample_rate == metadata.processed_sr


def test_postprocessor_no_export(preprocessed_data, tmp_path):
    """Test that export_files=False skips file creation."""
    # Arrange
    tensor, metadata = preprocessed_data
    post = Postprocessor()

    F, T = tensor.shape[2], tensor.shape[3]
    mock_output = torch.ones(1, 2, F, T) * 0.5

    # Act
    result = post.process(mock_output, metadata, tensor, tmp_path, "test", export_files=False)

    # Assert
    assert result.vocals_path is None
    assert result.instrumentals_path is None
    assert result.vocals_waveform is not None  # Waveforms still computed
    assert result.instrumentals_waveform is not None


def test_postprocessor_identity_mask_roundtrip(preprocessed_data, tmp_path):
    """Test that identity mask (all ones) approximately reconstructs original."""
    # Arrange
    tensor, metadata = preprocessed_data
    post = Postprocessor()

    # Identity mask = all ones (keep everything)
    F, T = tensor.shape[2], tensor.shape[3]
    identity_mask = torch.ones(1, 2, F, T)

    # Act
    result = post.process(identity_mask, metadata, tensor, tmp_path, "test")

    # Assert - waveform should not be silent (should have energy)
    vocals_energy = np.mean(result.vocals_waveform**2)
    assert vocals_energy > 1e-6, "Reconstructed audio is silent"


def test_postprocessor_zero_mask_produces_silence(preprocessed_data, tmp_path):
    """Test that zero mask produces near-silent output."""
    # Arrange
    tensor, metadata = preprocessed_data
    post = Postprocessor()

    # Zero mask = silence everything
    F, T = tensor.shape[2], tensor.shape[3]
    zero_mask = torch.zeros(1, 2, F, T)

    # Act
    result = post.process(zero_mask, metadata, tensor, tmp_path, "test")

    # Assert - waveform should be near-silent
    vocals_energy = np.mean(result.vocals_waveform**2)
    assert vocals_energy < 1e-6, "Zero mask should produce silence"


def test_postprocessor_named_stems_returns_four_stem_waveforms(preprocessed_data, tmp_path):
    """Test that four-stem output returns one waveform for each named stem."""
    # Arrange
    tensor, metadata = preprocessed_data
    post = Postprocessor()
    stems = ["drums", "bass", "vocals", "other"]

    F, T = tensor.shape[2], tensor.shape[3]
    mock_output = torch.full((1, len(stems), TARGET_CHANNELS, F, T), 0.25)

    # Act
    result = post.process(
        mock_output,
        metadata,
        tensor,
        tmp_path,
        "test",
        export_files=False,
        stems=stems,
    )

    # Assert
    assert set(result.stem_waveforms) == set(stems)
    assert result.stem_paths == {}
    assert result.vocals_waveform is result.stem_waveforms["vocals"]
    assert result.vocals_path is None
    assert result.instrumentals_waveform is None
    assert result.instrumentals_path is None

    for waveform in result.stem_waveforms.values():
        assert waveform.shape == (TARGET_CHANNELS, metadata.processed_length)


def test_postprocessor_named_stems_exports_files(preprocessed_data, tmp_path):
    """Test that named stems are exported using their stem names."""
    # Arrange
    tensor, metadata = preprocessed_data
    post = Postprocessor()
    stems = ["drums", "bass", "vocals", "other"]

    F, T = tensor.shape[2], tensor.shape[3]
    mock_output = torch.full((1, len(stems), TARGET_CHANNELS, F, T), 0.25)

    # Act
    result = post.process(
        mock_output,
        metadata,
        tensor,
        tmp_path,
        "song",
        export_files=True,
        stems=stems,
    )

    # Assert
    for stem in stems:
        assert result.stem_paths[stem] == tmp_path / f"song_{stem}.wav"
        assert result.stem_paths[stem].exists()

    assert result.vocals_path == tmp_path / "song_vocals.wav"
    assert result.vocals_waveform is result.stem_waveforms["vocals"]


def test_postprocessor_named_stems_preserves_instrumentals_fields(preprocessed_data, tmp_path):
    """Test that a named instrumentals stem fills convenience result fields."""
    # Arrange
    tensor, metadata = preprocessed_data
    post = Postprocessor()
    stems = ["vocals", "instrumentals"]

    F, T = tensor.shape[2], tensor.shape[3]
    mock_output = torch.full((1, len(stems), TARGET_CHANNELS, F, T), 0.5)

    # Act
    result = post.process(
        mock_output,
        metadata,
        tensor,
        tmp_path,
        "song",
        export_files=True,
        stems=stems,
    )

    # Assert
    assert result.vocals_path == tmp_path / "song_vocals.wav"
    assert result.instrumentals_path == tmp_path / "song_instrumentals.wav"
    assert result.vocals_waveform is result.stem_waveforms["vocals"]
    assert result.instrumentals_waveform is result.stem_waveforms["instrumentals"]


def test_postprocessor_named_stems_accepts_shared_channel_masks(preprocessed_data, tmp_path):
    """Test that [1, S, F, T] named-stem masks are converted to stereo masks."""
    # Arrange
    tensor, metadata = preprocessed_data
    post = Postprocessor()
    stems = ["vocals", "instrumentals"]

    F, T = tensor.shape[2], tensor.shape[3]
    mock_output = torch.full((1, len(stems), F, T), 0.5)

    # Act
    result = post.process(
        mock_output,
        metadata,
        tensor,
        tmp_path,
        "test",
        export_files=False,
        stems=stems,
    )

    # Assert
    assert result.vocals_waveform.shape == (TARGET_CHANNELS, metadata.processed_length)
    assert result.instrumentals_waveform.shape == (TARGET_CHANNELS, metadata.processed_length)


def test_postprocessor_rejects_empty_stem_list(preprocessed_data, tmp_path):
    """Test that named-stem processing rejects an empty stem list."""
    # Arrange
    tensor, metadata = preprocessed_data
    post = Postprocessor()
    F, T = tensor.shape[2], tensor.shape[3]
    mock_output = torch.ones(1, 1, TARGET_CHANNELS, F, T)

    # Act and Assert
    with pytest.raises(ValueError, match="stems cannot be empty"):
        post.process(mock_output, metadata, tensor, tmp_path, "test", stems=[])


def test_postprocessor_rejects_shared_mask_stem_count_mismatch(preprocessed_data, tmp_path):
    """Test that shared-channel masks must match the named stem count."""
    # Arrange
    tensor, metadata = preprocessed_data
    post = Postprocessor()
    F, T = tensor.shape[2], tensor.shape[3]
    mock_output = torch.ones(1, 1, F, T)

    # Act and Assert
    with pytest.raises(ValueError, match="Mask stem count"):
        post.process(
            mock_output,
            metadata,
            tensor,
            tmp_path,
            "test",
            stems=["vocals", "instrumentals"],
        )


def test_postprocessor_rejects_stereo_mask_stem_count_mismatch(preprocessed_data, tmp_path):
    """Test that stereo masks must match the named stem count."""
    # Arrange
    tensor, metadata = preprocessed_data
    post = Postprocessor()
    F, T = tensor.shape[2], tensor.shape[3]
    mock_output = torch.ones(1, 1, TARGET_CHANNELS, F, T)

    # Act and Assert
    with pytest.raises(ValueError, match="Mask stem count"):
        post.process(
            mock_output,
            metadata,
            tensor,
            tmp_path,
            "test",
            stems=["vocals", "instrumentals"],
        )


def test_postprocessor_rejects_named_stem_channel_count_mismatch(preprocessed_data, tmp_path):
    """Test that named-stem masks must contain the expected channel count."""
    # Arrange
    tensor, metadata = preprocessed_data
    post = Postprocessor()
    F, T = tensor.shape[2], tensor.shape[3]
    mock_output = torch.ones(1, 2, 1, F, T)

    # Act and Assert
    with pytest.raises(ValueError, match="Mask channel count"):
        post.process(
            mock_output,
            metadata,
            tensor,
            tmp_path,
            "test",
            stems=["vocals", "instrumentals"],
        )


def test_postprocessor_rejects_invalid_named_stem_output_dimensions(preprocessed_data, tmp_path):
    """Test that named-stem processing rejects unsupported tensor dimensions."""
    # Arrange
    tensor, metadata = preprocessed_data
    post = Postprocessor()
    F, T = tensor.shape[2], tensor.shape[3]
    mock_output = torch.ones(1, 2, TARGET_CHANNELS, F, T, 1)

    # Act and Assert
    with pytest.raises(ValueError, match="model_output must have shape"):
        post.process(
            mock_output,
            metadata,
            tensor,
            tmp_path,
            "test",
            stems=["vocals", "instrumentals"],
        )


def test_postprocessor_transposes_channel_last_stem_waveform(
    preprocessed_data,
    tmp_path,
    monkeypatch,
):
    """Test that named-stem output transposes a channel-last waveform."""
    # Arrange
    tensor, metadata = preprocessed_data
    post = Postprocessor()
    F, T = tensor.shape[2], tensor.shape[3]
    mock_output = torch.ones(1, 1, TARGET_CHANNELS, F, T)

    def fake_reconstruct_stem(_mask, _original_magnitude, reconstruction_metadata):
        return np.ones(
            (reconstruction_metadata.processed_length, TARGET_CHANNELS),
            dtype=np.float32,
        )

    monkeypatch.setattr(post, "_reconstruct_stem", fake_reconstruct_stem)

    # Act
    result = post.process(
        mock_output,
        metadata,
        tensor,
        tmp_path,
        "test",
        export_files=False,
        stems=["vocals"],
    )

    # Assert
    assert result.vocals_waveform.shape == (TARGET_CHANNELS, metadata.processed_length)


def test_postprocessor_rejects_invalid_reconstructed_stem_shape(
    preprocessed_data,
    tmp_path,
    monkeypatch,
):
    """Test that named-stem processing rejects a malformed waveform result."""
    # Arrange
    tensor, metadata = preprocessed_data
    post = Postprocessor()
    F, T = tensor.shape[2], tensor.shape[3]
    mock_output = torch.ones(1, 1, TARGET_CHANNELS, F, T)

    def fake_reconstruct_stem(_mask, _original_magnitude, reconstruction_metadata):
        return np.ones(reconstruction_metadata.processed_length, dtype=np.float32)

    monkeypatch.setattr(post, "_reconstruct_stem", fake_reconstruct_stem)

    # Act and Assert
    with pytest.raises(ValueError, match="Reconstructed waveform has wrong shape"):
        post.process(
            mock_output,
            metadata,
            tensor,
            tmp_path,
            "test",
            export_files=False,
            stems=["vocals"],
        )


def test_postprocessor_fallback_duplicates_single_channel_magnitude(preprocessed_data, tmp_path):
    """Test fallback reconstruction from a single-channel input tensor."""
    # Arrange
    tensor, metadata = preprocessed_data
    post = Postprocessor()
    fallback_metadata = metadata_without_mix_magnitude(metadata)

    F, T = tensor.shape[2], tensor.shape[3]
    mono_input = torch.ones(1, 1, F, T)
    mock_output = torch.ones(1, 2, F, T)

    # Act
    result = post.process(
        mock_output,
        fallback_metadata,
        mono_input,
        tmp_path,
        "test",
        export_files=False,
    )

    # Assert
    assert result.vocals_waveform.shape == (TARGET_CHANNELS, metadata.processed_length)


def test_postprocessor_fallback_duplicates_two_dimensional_magnitude(
    preprocessed_data,
    tmp_path,
):
    """Test fallback reconstruction from a two-dimensional input tensor."""
    # Arrange
    tensor, metadata = preprocessed_data
    post = Postprocessor()
    fallback_metadata = metadata_without_mix_magnitude(metadata)

    F, T = tensor.shape[2], tensor.shape[3]
    two_dimensional_input = torch.ones(F, T)
    mock_output = torch.ones(1, 2, F, T)

    # Act
    result = post.process(
        mock_output,
        fallback_metadata,
        two_dimensional_input,
        tmp_path,
        "test",
        export_files=False,
    )

    # Assert
    assert result.instrumentals_waveform.shape == (
        TARGET_CHANNELS,
        metadata.processed_length,
    )


def test_postprocessor_rejects_invalid_original_magnitude_shape(preprocessed_data, tmp_path):
    """Test that processing rejects stored magnitude data without two channels."""
    # Arrange
    tensor, metadata = preprocessed_data
    post = Postprocessor()
    F, T = tensor.shape[2], tensor.shape[3]
    invalid_metadata = SimpleNamespace(mix_magnitude=np.ones((1, F, T), dtype=np.float32))
    mock_output = torch.ones(1, 2, F, T)

    # Act and Assert
    with pytest.raises(ValueError, match="original_magnitude must have shape"):
        post.process(mock_output, invalid_metadata, tensor, tmp_path, "test")


def test_postprocessor_rejects_invalid_legacy_model_output(preprocessed_data, tmp_path):
    """Test that legacy two-stem mode rejects an invalid output shape."""
    # Arrange
    tensor, metadata = preprocessed_data
    post = Postprocessor()
    F, T = tensor.shape[2], tensor.shape[3]
    invalid_output = torch.ones(1, 3, F, T)

    # Act and Assert
    with pytest.raises(ValueError, match="model_output must have shape"):
        post.process(invalid_output, metadata, tensor, tmp_path, "test")


def test_reconstruct_stem_rejects_invalid_mask_shape(preprocessed_data):
    """Test that stem reconstruction rejects a mask without two channels."""
    # Arrange
    tensor, metadata = preprocessed_data
    post = Postprocessor()
    F, T = tensor.shape[2], tensor.shape[3]
    invalid_mask = np.ones((1, F, T), dtype=np.float32)
    original_magnitude = np.ones((TARGET_CHANNELS, F, T), dtype=np.float32)

    # Act and Assert
    with pytest.raises(ValueError, match="mask must have shape"):
        post._reconstruct_stem(invalid_mask, original_magnitude, metadata)


def test_reconstruct_stem_rejects_invalid_original_magnitude_shape(preprocessed_data):
    """Test that stem reconstruction rejects magnitude data without two channels."""
    # Arrange
    tensor, metadata = preprocessed_data
    post = Postprocessor()
    F, T = tensor.shape[2], tensor.shape[3]
    mask = np.ones((TARGET_CHANNELS, F, T), dtype=np.float32)
    invalid_magnitude = np.ones((1, F, T), dtype=np.float32)

    # Act and Assert
    with pytest.raises(ValueError, match="original_magnitude must have shape"):
        post._reconstruct_stem(mask, invalid_magnitude, metadata)
