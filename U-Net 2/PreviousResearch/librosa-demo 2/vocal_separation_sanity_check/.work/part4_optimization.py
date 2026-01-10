# ============================================
# OPTIMIZATION
# ============================================

def optimize_one_slice(slice_name, vocal_fps, mixture_mag, num_windows, sr, current_mag=None, iteration_offset=0):
    """
    Optimize EQ curves for ONE slice using gradient descent.

    Args:
        slice_name: Name of the slice to optimize (e.g., 'slice_1_horizontal')
        vocal_fps: Vocal fingerprints dict
        mixture_mag: Original mixture magnitude spectrogram
        num_windows: Number of time windows
        sr: Sample rate
        current_mag: Current working magnitude (for sequential refinement). If None, starts from mixture_mag
        iteration_offset: Display offset for iteration counter (for multi-pass clarity)

    Returns:
        tuple: (refined_magnitude, losses)
            - refined_magnitude: Updated magnitude after applying learned EQ
            - losses: List of loss values per iteration
    """

    print(f"\n{'='*70}")
    print(f"OPTIMIZING: {slice_name}")
    print(f"{'='*70}")
    print(f"\nOptimizing {num_windows} windows × 400 EQ points...")

    # Use current_mag if provided (sequential), otherwise start from mixture
    if current_mag is None:
        working_mag = mixture_mag.copy()
        print(f"  Starting from: original mixture")
    else:
        working_mag = current_mag.copy()
        print(f"  Starting from: previous pass result")

    # Initialize EQ curves (start with unity gain)
    eq_curves = [np.ones(400) for _ in range(num_windows)]

    # Track loss
    losses = []

    # Optimization loop
    for iteration in range(CONFIG['num_iterations']):
        total_loss = 0.0

        # Process each window
        for win_idx in range(num_windows):
            # Get target vocal fingerprint for this slice
            vocal_fp = vocal_fps[slice_name][win_idx]['freq_profile_400']

            # Get current working magnitude window
            mixture_window = working_mag[:, win_idx]

            # Convert to 400-point representation
            mixture_fp = np.interp(
                x=np.linspace(0, sr/2, 400),
                xp=np.linspace(0, sr/2, len(mixture_window)),
                fp=mixture_window
            )

            # Apply current EQ
            adjusted_fp = mixture_fp * eq_curves[win_idx]

            # Compute loss (mean squared error)
            loss = np.mean((adjusted_fp - vocal_fp)**2)
            total_loss += loss

            # Compute gradient
            gradient = 2 * (adjusted_fp - vocal_fp) * mixture_fp

            # Update EQ curve (gradient descent)
            eq_curves[win_idx] -= CONFIG['learning_rate'] * gradient

            # Clip to reasonable range [0.1, 3.0]
            eq_curves[win_idx] = np.clip(eq_curves[win_idx], 0.1, 3.0)

        avg_loss = total_loss / num_windows
        losses.append(avg_loss)

        if iteration % 20 == 0:
            display_iter = iteration_offset + iteration
            print(f"  Iteration {display_iter:3d}: Loss = {avg_loss:.6f}")

    print(f"\n✓ Optimization complete!")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Initial loss: {losses[0]:.6f}")
    print(f"  Improvement: {(1 - losses[-1]/losses[0])*100:.1f}%")

    # Apply learned EQ to working magnitude
    refined_mag = np.zeros_like(working_mag)

    for win_idx in range(num_windows):
        window_mag = working_mag[:, win_idx]

        # Interpolate 400-point EQ to full STFT bins
        freq_bins_stft = np.linspace(0, sr/2, len(window_mag))
        freq_points_eq = np.linspace(0, sr/2, 400)

        eq_curve_full = np.interp(freq_bins_stft, freq_points_eq, eq_curves[win_idx])

        # Apply EQ (multiplicative)
        refined_mag[:, win_idx] = window_mag * eq_curve_full

    print(f"  ✓ EQ curves applied to magnitude")

    return refined_mag, losses

def reconstruct_audio_from_magnitude(magnitude, phase, hop_length, n_fft):
    """
    Reconstruct time-domain audio from magnitude spectrogram and phase.

    Args:
        magnitude: Magnitude spectrogram
        phase: Phase spectrogram
        hop_length: STFT hop length
        n_fft: STFT FFT size

    Returns:
        numpy.ndarray: Reconstructed audio (normalized)
    """

    # Reconstruct complex STFT
    reconstructed_stft = magnitude * np.exp(1j * phase)

    # Inverse STFT
    audio = librosa.istft(
        reconstructed_stft,
        hop_length=hop_length,
        n_fft=n_fft
    )

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-8)

    return audio
