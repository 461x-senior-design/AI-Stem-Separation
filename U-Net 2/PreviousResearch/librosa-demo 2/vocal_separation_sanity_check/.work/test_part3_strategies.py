# ============================================
# STRATEGY 1: NORMALIZE AFTER EACH PASS
# ============================================

def optimize_strategy1_normalize(slice_name, vocal_fps, mixture_mag, num_windows, sr, current_mag=None):
    """
    Strategy 1: Normalize after each pass to prevent cumulative gain loss
    """
    if current_mag is None:
        working_mag = mixture_mag.copy()
    else:
        working_mag = current_mag.copy()

    # Store original max for normalization reference
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

    # STRATEGY 1: Normalize to maintain consistent level
    current_max = np.max(refined_mag)
    if current_max > 0:
        refined_mag = refined_mag / current_max * original_max

    return refined_mag


# ============================================
# STRATEGY 2: TRACK TOTAL GAIN AND COMPENSATE
# ============================================

def optimize_strategy2_compensate(slice_name, vocal_fps, mixture_mag, num_windows, sr, current_mag=None, accumulated_gains=None):
    """
    Strategy 2: Track cumulative gain changes and compensate at the end
    Returns (refined_mag, accumulated_gains)
    """
    if current_mag is None:
        working_mag = mixture_mag.copy()
    else:
        working_mag = current_mag.copy()

    if accumulated_gains is None:
        accumulated_gains = [np.ones(400) for _ in range(num_windows)]

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

        # STRATEGY 2: Accumulate gains
        eq_curve_400 = eq_curves[win_idx]
        accumulated_gains[win_idx] = accumulated_gains[win_idx] * eq_curve_400

    return refined_mag, accumulated_gains


# ============================================
# STRATEGY 3: DB-SPACE EQ (ADDITIVE)
# ============================================

def optimize_strategy3_db_space(slice_name, vocal_fps, mixture_mag, num_windows, sr, current_mag=None):
    """
    Strategy 3: Work in dB space (additive) instead of linear space (multiplicative)
    """
    if current_mag is None:
        working_mag = mixture_mag.copy()
    else:
        working_mag = current_mag.copy()

    # Convert to dB using librosa (safer, handles edge cases better)
    working_mag_db = librosa.amplitude_to_db(working_mag, ref=1.0, amin=1e-8)

    # Initialize EQ curves (dB adjustments, start at 0)
    eq_curves_db = [np.zeros(400) for _ in range(num_windows)]

    # Optimization loop
    for iteration in range(CONFIG['num_iterations']):
        total_loss = 0.0

        for win_idx in range(num_windows):
            vocal_fp = vocal_fps[slice_name][win_idx]['freq_profile_400']
            # Convert target to dB using librosa
            vocal_fp_db = librosa.amplitude_to_db(vocal_fp, ref=1.0, amin=1e-8)

            mixture_window_db = working_mag_db[:, win_idx]

            mixture_fp_db = np.interp(
                x=np.linspace(0, sr/2, 400),
                xp=np.linspace(0, sr/2, len(mixture_window_db)),
                fp=mixture_window_db
            )

            # Apply EQ (additive in dB space)
            adjusted_fp_db = mixture_fp_db + eq_curves_db[win_idx]

            # Loss in dB space
            loss = np.mean((adjusted_fp_db - vocal_fp_db)**2)
            total_loss += loss

            # Gradient (simpler in dB space - additive)
            gradient = 2 * (adjusted_fp_db - vocal_fp_db)

            # Update EQ curve
            eq_curves_db[win_idx] -= CONFIG['learning_rate'] * gradient

            # STRATEGY 3: Clip in dB space (symmetric)
            eq_curves_db[win_idx] = np.clip(eq_curves_db[win_idx], -20, 20)  # ±20dB

    # Apply EQ in dB space
    refined_mag_db = np.zeros_like(working_mag_db)
    for win_idx in range(num_windows):
        window_mag_db = working_mag_db[:, win_idx]
        freq_bins_stft = np.linspace(0, sr/2, len(window_mag_db))
        freq_points_eq = np.linspace(0, sr/2, 400)
        eq_curve_db_full = np.interp(freq_bins_stft, freq_points_eq, eq_curves_db[win_idx])
        refined_mag_db[:, win_idx] = window_mag_db + eq_curve_db_full

    # Convert back to linear using librosa
    refined_mag = librosa.db_to_amplitude(refined_mag_db, ref=1.0)
    refined_mag = np.nan_to_num(refined_mag, nan=0.0, posinf=0.0, neginf=0.0)

    return refined_mag


# ============================================
# STRATEGY 4: SYMMETRIC CLIPPING RANGE
# ============================================

def optimize_strategy4_symmetric(slice_name, vocal_fps, mixture_mag, num_windows, sr, current_mag=None):
    """
    Strategy 4: Use symmetric clipping range [0.333, 3.0] = [-10dB, +10dB]
    Instead of [0.1, 3.0] = [-20dB, +9.5dB]
    """
    if current_mag is None:
        working_mag = mixture_mag.copy()
    else:
        working_mag = current_mag.copy()

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

            # STRATEGY 4: Symmetric clipping in dB space
            # [0.333, 3.0] ≈ [-10dB, +10dB]
            eq_curves[win_idx] = np.clip(eq_curves[win_idx], 0.333, 3.0)

    # Apply EQ
    refined_mag = np.zeros_like(working_mag)
    for win_idx in range(num_windows):
        window_mag = working_mag[:, win_idx]
        freq_bins_stft = np.linspace(0, sr/2, len(window_mag))
        freq_points_eq = np.linspace(0, sr/2, 400)
        eq_curve_full = np.interp(freq_bins_stft, freq_points_eq, eq_curves[win_idx])
        refined_mag[:, win_idx] = window_mag * eq_curve_full

    return refined_mag
