# ============================================
# FINGERPRINT EXTRACTION
# ============================================

def window_to_bottleneck(window, sr):
    """
    Compress window through encoder layers to bottleneck.
    Extract 425 metrics: 400-point frequency profile + 25 derived features.

    Args:
        window: 1D frequency spectrum from one time window
        sr: Sample rate

    Returns:
        dict: Fingerprint metrics for this window
    """

    # Suppress warnings for log operations
    with np.errstate(divide='ignore', invalid='ignore'):
        # Compress through encoder layers (5 downsample steps)
        layer1 = downsample_spectrum(window, factor=2)
        layer2 = downsample_spectrum(layer1, factor=2)
        layer3 = downsample_spectrum(layer2, factor=2)
        layer4 = downsample_spectrum(layer3, factor=2)
        bottleneck_vector = downsample_spectrum(layer4, factor=2)

        # Core: 400-point frequency profile
        freq_profile_400 = np.interp(
            x=np.linspace(0, sr/2, 400),
            xp=np.linspace(0, sr/2, len(layer2)),
            fp=layer2
        )

        # Band energies (6 frequency bands)
        num_bins = len(layer2)
        bass_bins = slice(0, max(1, int(num_bins * 250/(sr/2))))
        low_mid_bins = slice(max(1, int(num_bins * 250/(sr/2))), int(num_bins * 500/(sr/2)))
        mid_bins = slice(int(num_bins * 500/(sr/2)), int(num_bins * 2000/(sr/2)))
        high_mid_bins = slice(int(num_bins * 2000/(sr/2)), int(num_bins * 4000/(sr/2)))
        presence_bins = slice(int(num_bins * 4000/(sr/2)), min(num_bins, int(num_bins * 8000/(sr/2))))
        high_bins = slice(int(num_bins * 8000/(sr/2)), num_bins)

        bass_energy = np.sum(layer2[bass_bins]**2) + 1e-8
        low_mid_energy = np.sum(layer2[low_mid_bins]**2) + 1e-8
        mid_energy = np.sum(layer2[mid_bins]**2) + 1e-8
        high_mid_energy = np.sum(layer2[high_mid_bins]**2) + 1e-8
        presence_energy = np.sum(layer2[presence_bins]**2) + 1e-8
        high_energy = np.sum(layer2[high_bins]**2) + 1e-8

        # Spectral shape features
        freqs_l4 = np.linspace(0, sr/2, len(layer4))
        centroid = np.sum(freqs_l4 * layer4) / (np.sum(layer4) + 1e-8)
        spread = np.sqrt(np.sum(((freqs_l4 - centroid)**2) * layer4) / (np.sum(layer4) + 1e-8))

        cumsum_energy = np.cumsum(layer4)
        total = cumsum_energy[-1]
        rolloff_idx = np.where(cumsum_energy >= 0.85 * total)[0]
        rolloff = freqs_l4[rolloff_idx[0]] if len(rolloff_idx) > 0 else sr/2

        geo_mean = np.exp(np.mean(np.log(layer4 + 1e-8)))
        flatness = geo_mean / (np.mean(layer4) + 1e-8)
        slope = (layer4[-1] - layer4[0]) / len(layer4)
        crest = np.max(layer4) / (np.mean(layer4) + 1e-8)

        # Harmonic structure
        peaks, properties = find_peaks(layer3, height=np.max(layer3)*0.1)
        num_harmonics = len(peaks)

        if num_harmonics > 1:
            harmonic_spacing = np.mean(np.diff(peaks)) * (sr/2) / len(layer3)
            fundamental = harmonic_spacing
        else:
            harmonic_spacing = 0
            fundamental = 0

        harmonic_strength = np.mean(properties['peak_heights']) / (np.mean(layer3) + 1e-8) if num_harmonics > 0 else 0

        # Formant detection (vocal-specific features)
        mid_range_peaks, _ = find_peaks(layer2[mid_bins], height=np.max(layer2[mid_bins])*0.3)
        formants = []
        for peak_idx in mid_range_peaks[:3]:
            formant_freq = (mid_bins.start + peak_idx) * (sr/2) / len(layer2)
            formants.append(formant_freq)
        while len(formants) < 3:
            formants.append(0)

        formant_strength = np.mean([layer2[mid_bins.start + p] for p in mid_range_peaks]) if len(mid_range_peaks) > 0 else 0

        # Dynamics features
        peak_to_rms = np.max(layer4) / (np.sqrt(np.mean(layer4**2)) + 1e-8)
        top_10_percent = int(len(layer4) * 0.1)
        top_energy = np.sum(np.sort(layer4)[-top_10_percent:])
        energy_concentration = top_energy / (np.sum(layer4) + 1e-8)

        normalized = layer4 / (np.sum(layer4) + 1e-8)
        entropy = -np.sum(normalized * np.log2(normalized + 1e-8))
        total_energy = np.sum(bottleneck_vector**2)

        return {
            'freq_profile_400': np.nan_to_num(freq_profile_400, nan=0.0, posinf=0.0, neginf=0.0),
            'bass_energy': bass_energy,
            'low_mid_energy': low_mid_energy,
            'mid_energy': mid_energy,
            'high_mid_energy': high_mid_energy,
            'presence_energy': presence_energy,
            'high_energy': high_energy,
            'mid_to_bass_ratio': mid_energy / bass_energy,
            'high_to_mid_ratio': high_energy / mid_energy,
            'spectral_centroid': centroid,
            'spectral_spread': spread if not np.isnan(spread) else 0.0,
            'spectral_rolloff': rolloff,
            'spectral_flatness': flatness if not np.isnan(flatness) else 0.0,
            'spectral_slope': slope,
            'spectral_crest': crest,
            'fundamental_frequency': fundamental,
            'num_harmonics': num_harmonics,
            'harmonic_spacing': harmonic_spacing,
            'harmonic_strength': harmonic_strength if not np.isnan(harmonic_strength) else 0.0,
            'formant_1': formants[0],
            'formant_2': formants[1],
            'formant_3': formants[2],
            'formant_strength': formant_strength if not np.isnan(formant_strength) else 0.0,
            'peak_to_rms': peak_to_rms,
            'energy_concentration': energy_concentration,
            'spectral_entropy': entropy if not np.isnan(entropy) else 0.0,
            'total_energy': total_energy,
        }

def process_audio_to_fingerprints(audio_path, sr, n_fft, hop_length):
    """
    Load audio file and create complete spectral fingerprints.

    Args:
        audio_path: Path to audio file
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length for STFT

    Returns:
        tuple: (audio, stft, magnitude, fingerprints)
            - audio: Time-domain waveform
            - stft: Complex STFT
            - magnitude: Magnitude spectrogram
            - fingerprints: Dict of {slice_name: [window_fingerprints]}
    """

    print(f"\n[Processing: {Path(audio_path).name}]")

    # Load audio
    audio, _ = librosa.load(audio_path, sr=sr, duration=CONFIG['duration'])
    print(f"  Loaded {len(audio)} samples")

    # Create STFT
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    num_windows = magnitude.shape[1]
    print(f"  Spectrogram: {magnitude.shape} ({num_windows} windows)")

    # Create 18 slices
    print(f"  Creating 18 spectral slices...")
    slices = create_18_slices(magnitude)

    # Process each slice to bottleneck fingerprints
    fingerprints = {}

    for slice_name, slice_data in slices.items():
        slice_fingerprints = []
        num_slice_windows = slice_data.shape[1]

        for window_idx in range(num_slice_windows):
            window = slice_data[:, window_idx]
            metrics = window_to_bottleneck(window, sr)
            slice_fingerprints.append(metrics)

        fingerprints[slice_name] = slice_fingerprints

    print(f"  ✓ Created fingerprints for {len(fingerprints)} slices")

    return audio, stft, magnitude, fingerprints
