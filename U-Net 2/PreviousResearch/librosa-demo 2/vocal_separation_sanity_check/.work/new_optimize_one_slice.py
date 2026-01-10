def optimize_one_slice(slice_name, vocal_fps, mixture_mag, num_windows, sr, current_mag=None, iteration_offset=0):
    """
    Optimize EQ curves for ONE slice using gradient descent in dB-space.

    Uses Strategy 3 (dB-space EQ) from test_gain_strategies.py for numerical stability.

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

    # Convert to dB space using librosa (Strategy 3: dB-space optimization)
    working_mag_db = librosa.amplitude_to_db(working_mag, ref=1.0, amin=1e-8)

    # Initialize EQ curves (dB adjustments, start at 0)
    eq_curves_db = [np.zeros(400) for _ in range(num_windows)]

    # Track loss
    losses = []

    # Optimization loop
    for iteration in range(CONFIG['num_iterations']):
        total_loss = 0.0

        # Process each window
        for win_idx in range(num_windows):
            # Get target vocal fingerprint for this slice
            vocal_fp = vocal_fps[slice_name][win_idx]['freq_profile_400']
            # Convert target to dB using librosa
            vocal_fp_db = librosa.amplitude_to_db(vocal_fp, ref=1.0, amin=1e-8)

            # Get current working magnitude window (in dB space)
            mixture_window_db = working_mag_db[:, win_idx]

            # Convert to 400-point representation
            mixture_fp_db = np.interp(
                x=np.linspace(0, sr/2, 400),
                xp=np.linspace(0, sr/2, len(mixture_window_db)),
                fp=mixture_window_db
            )

            # Apply current EQ (additive in dB space)
            adjusted_fp_db = mixture_fp_db + eq_curves_db[win_idx]

            # Compute loss (mean squared error in dB space)
            loss = np.mean((adjusted_fp_db - vocal_fp_db)**2)
            total_loss += loss

            # Compute gradient (simpler in dB space - additive)
            gradient = 2 * (adjusted_fp_db - vocal_fp_db)

            # Update EQ curve (gradient descent)
            eq_curves_db[win_idx] -= CONFIG['learning_rate'] * gradient

            # Clip to symmetric dB range [±20 dB]
            eq_curves_db[win_idx] = np.clip(eq_curves_db[win_idx], -20, 20)

        avg_loss = total_loss / num_windows
        losses.append(avg_loss)

        if iteration % 20 == 0:
            display_iter = iteration_offset + iteration
            print(f"  Iteration {display_iter:3d}: Loss = {avg_loss:.6f}")

    print(f"\n✓ Optimization complete!")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Initial loss: {losses[0]:.6f}")
    print(f"  Improvement: {(1 - losses[-1]/losses[0])*100:.1f}%")

    # Apply learned EQ to working magnitude (in dB space)
    refined_mag_db = np.zeros_like(working_mag_db)

    for win_idx in range(num_windows):
        window_mag_db = working_mag_db[:, win_idx]

        # Interpolate 400-point EQ to full STFT bins
        freq_bins_stft = np.linspace(0, sr/2, len(window_mag_db))
        freq_points_eq = np.linspace(0, sr/2, 400)

        eq_curve_db_full = np.interp(freq_bins_stft, freq_points_eq, eq_curves_db[win_idx])

        # Apply EQ (additive in dB space)
        refined_mag_db[:, win_idx] = window_mag_db + eq_curve_db_full

    # Convert back to linear amplitude using librosa
    refined_mag = librosa.db_to_amplitude(refined_mag_db, ref=1.0)
    refined_mag = np.nan_to_num(refined_mag, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  ✓ EQ curves applied in dB space, converted back to linear")

    return refined_mag, losses
