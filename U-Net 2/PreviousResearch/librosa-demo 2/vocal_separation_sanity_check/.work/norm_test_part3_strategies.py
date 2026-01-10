# ============================================
# NORMALIZATION STRATEGIES
# ============================================

def optimize_with_normalization(slice_name, vocal_fps, mixture_mag, num_windows, sr, current_mag, norm_strategy):
    """
    Optimize with specified normalization strategy.

    Args:
        norm_strategy: One of 'manual', 'librosa_global', 'librosa_axis0', 'librosa_axis1'
    """
    if current_mag is None:
        working_mag = mixture_mag.copy()
    else:
        working_mag = current_mag.copy()

    # Store original max for manual normalization
    original_max = np.max(working_mag)

    # Initialize EQ curves
    eq_curves = [np.ones(400) for _ in range(num_windows)]

    # Optimization loop
    for iteration in range(CONFIG['num_iterations']):
        total_loss = 0.0

        for win_idx in range(num_windows):
            vocal_fp = vocal_fps[slice_name][win_idx]['freq_profile_400']
            mixture_window = working_mag[:, win_idx]

            mixture_fp = np.interp(
                x=np.linspace(0, sr/2, 400),
                xp=np.linspace(0, sr/2, len(mixture_window)),
                fp=mixture_window
            )

            adjusted_fp = mixture_fp * eq_curves[win_idx]
            loss = np.mean((adjusted_fp - vocal_fp)**2)
            total_loss += loss

            gradient = 2 * (adjusted_fp - vocal_fp) * mixture_fp
            eq_curves[win_idx] -= CONFIG['learning_rate'] * gradient
            eq_curves[win_idx] = np.clip(eq_curves[win_idx], 0.1, 3.0)

    # Apply EQ
    refined_mag = np.zeros_like(working_mag)
    for win_idx in range(num_windows):
        window_mag = working_mag[:, win_idx]
        freq_bins_stft = np.linspace(0, sr/2, len(window_mag))
        freq_points_eq = np.linspace(0, sr/2, 400)
        eq_curve_full = np.interp(freq_bins_stft, freq_points_eq, eq_curves[win_idx])
        refined_mag[:, win_idx] = window_mag * eq_curve_full

    # APPLY NORMALIZATION STRATEGY
    if norm_strategy == 'manual':
        # Strategy 1: Manual global normalization
        current_max = np.max(refined_mag)
        if current_max > 0:
            refined_mag = refined_mag / current_max * original_max

    elif norm_strategy == 'librosa_global':
        # Strategy 2: librosa.util.normalize (global, default)
        refined_mag = librosa.util.normalize(refined_mag) * original_max

    elif norm_strategy == 'librosa_axis0':
        # Strategy 3: librosa.util.normalize per-frequency bin (axis=0)
        # Normalize each frequency bin independently, then scale to original max
        refined_mag = librosa.util.normalize(refined_mag, axis=0) * original_max

    elif norm_strategy == 'librosa_axis1':
        # Strategy 4: librosa.util.normalize per-time window (axis=1)
        # Normalize each time window independently, then scale to original max
        refined_mag = librosa.util.normalize(refined_mag, axis=1) * original_max

    return refined_mag
