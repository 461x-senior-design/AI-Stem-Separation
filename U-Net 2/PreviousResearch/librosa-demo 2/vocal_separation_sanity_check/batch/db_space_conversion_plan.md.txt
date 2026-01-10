# DB-SPACE CONVERSION PLAN

## Goal
Replace `optimize_one_slice()` in `sanity_check_progressive.py` with EXACT Strategy 3 (dB-space) logic from `test_gain_strategies.py`

**Result:** Slice 5 in progressive should sound IDENTICAL to dB-space output from test_gain_strategies

## Line-by-Line Comparison

### CURRENT: optimize_one_slice() - LINEAR SPACE
```python
# Line 426-430: Working magnitude (LINEAR)
if current_mag is None:
    working_mag = mixture_mag.copy()
else:
    working_mag = current_mag.copy()

# Line 433: Initialize EQ curves (MULTIPLICATIVE - starts at 1.0)
eq_curves = [np.ones(400) for _ in range(num_windows)]

# Line 445: Target vocal (LINEAR)
vocal_fp = vocal_fps[slice_name][win_idx]['freq_profile_400']

# Line 448-455: Mixture window (LINEAR)
mixture_window = working_mag[:, win_idx]
mixture_fp = np.interp(
    x=np.linspace(0, sr/2, 400),
    xp=np.linspace(0, sr/2, len(mixture_window)),
    fp=mixture_window
)

# Line 458: Apply EQ (MULTIPLICATIVE)
adjusted_fp = mixture_fp * eq_curves[win_idx]

# Line 461: Loss (LINEAR SPACE)
loss = np.mean((adjusted_fp - vocal_fp)**2)

# Line 465: Gradient (MULTIPLICATIVE - includes mixture_fp factor)
gradient = 2 * (adjusted_fp - vocal_fp) * mixture_fp

# Line 468: Update
eq_curves[win_idx] -= CONFIG['learning_rate'] * gradient

# Line 471: Clip (LINEAR RANGE)
eq_curves[win_idx] = np.clip(eq_curves[win_idx], 0.1, 3.0)

# Line 486-498: Apply EQ (MULTIPLICATIVE)
refined_mag = np.zeros_like(working_mag)
for win_idx in range(num_windows):
    window_mag = working_mag[:, win_idx]
    eq_curve_full = np.interp(freq_bins_stft, freq_points_eq, eq_curves[win_idx])
    refined_mag[:, win_idx] = window_mag * eq_curve_full
```

### TARGET: optimize_strategy3_db_space() - DB SPACE
```python
# Line 283-286: Working magnitude (CONVERT TO DB!)
if current_mag is None:
    working_mag = mixture_mag.copy()
else:
    working_mag = current_mag.copy()

# NEW: Convert to dB
working_mag_db = librosa.amplitude_to_db(working_mag, ref=1.0, amin=1e-8)

# Line 292: Initialize EQ curves (ADDITIVE - starts at 0 dB!)
eq_curves_db = [np.zeros(400) for _ in range(num_windows)]

# Line 299: Target vocal (CONVERT TO DB!)
vocal_fp = vocal_fps[slice_name][win_idx]['freq_profile_400']
vocal_fp_db = librosa.amplitude_to_db(vocal_fp, ref=1.0, amin=1e-8)

# Line 303-309: Mixture window (DB SPACE!)
mixture_window_db = working_mag_db[:, win_idx]
mixture_fp_db = np.interp(
    x=np.linspace(0, sr/2, 400),
    xp=np.linspace(0, sr/2, len(mixture_window_db)),
    fp=mixture_window_db
)

# Line 312: Apply EQ (ADDITIVE!)
adjusted_fp_db = mixture_fp_db + eq_curves_db[win_idx]

# Line 315: Loss (DB SPACE)
loss = np.mean((adjusted_fp_db - vocal_fp_db)**2)

# Line 319: Gradient (ADDITIVE - simpler, no mixture_fp factor!)
gradient = 2 * (adjusted_fp_db - vocal_fp_db)

# Line 322: Update
eq_curves_db[win_idx] -= CONFIG['learning_rate'] * gradient

# Line 325: Clip (DB RANGE ±20 dB)
eq_curves_db[win_idx] = np.clip(eq_curves_db[win_idx], -20, 20)

# Line 328-337: Apply EQ (ADDITIVE IN DB SPACE!)
refined_mag_db = np.zeros_like(working_mag_db)
for win_idx in range(num_windows):
    window_mag_db = working_mag_db[:, win_idx]
    eq_curve_db_full = np.interp(freq_bins_stft, freq_points_eq, eq_curves_db[win_idx])
    refined_mag_db[:, win_idx] = window_mag_db + eq_curve_db_full

# Line 337-338: Convert back to LINEAR!
refined_mag = librosa.db_to_amplitude(refined_mag_db, ref=1.0)
refined_mag = np.nan_to_num(refined_mag, nan=0.0, posinf=0.0, neginf=0.0)
```

## KEY CHANGES NEEDED

### 1. After line 430 (after setting working_mag):
```python
# Convert to dB space using librosa
working_mag_db = librosa.amplitude_to_db(working_mag, ref=1.0, amin=1e-8)
```

### 2. Replace line 433:
```python
# OLD:
eq_curves = [np.ones(400) for _ in range(num_windows)]

# NEW:
eq_curves_db = [np.zeros(400) for _ in range(num_windows)]
```

### 3. Inside optimization loop, after line 445 (after getting vocal_fp):
```python
# Convert target to dB
vocal_fp_db = librosa.amplitude_to_db(vocal_fp, ref=1.0, amin=1e-8)
```

### 4. Replace lines 448-455 (mixture window extraction):
```python
# OLD:
mixture_window = working_mag[:, win_idx]
mixture_fp = np.interp(...)

# NEW:
mixture_window_db = working_mag_db[:, win_idx]
mixture_fp_db = np.interp(
    x=np.linspace(0, sr/2, 400),
    xp=np.linspace(0, sr/2, len(mixture_window_db)),
    fp=mixture_window_db
)
```

### 5. Replace line 458 (apply EQ):
```python
# OLD:
adjusted_fp = mixture_fp * eq_curves[win_idx]

# NEW:
adjusted_fp_db = mixture_fp_db + eq_curves_db[win_idx]
```

### 6. Replace line 461 (loss):
```python
# OLD:
loss = np.mean((adjusted_fp - vocal_fp)**2)

# NEW:
loss = np.mean((adjusted_fp_db - vocal_fp_db)**2)
```

### 7. Replace line 465 (gradient):
```python
# OLD:
gradient = 2 * (adjusted_fp - vocal_fp) * mixture_fp

# NEW:
gradient = 2 * (adjusted_fp_db - vocal_fp_db)
```

### 8. Replace line 468 (update):
```python
# OLD:
eq_curves[win_idx] -= CONFIG['learning_rate'] * gradient

# NEW:
eq_curves_db[win_idx] -= CONFIG['learning_rate'] * gradient
```

### 9. Replace line 471 (clip):
```python
# OLD:
eq_curves[win_idx] = np.clip(eq_curves[win_idx], 0.1, 3.0)

# NEW:
eq_curves_db[win_idx] = np.clip(eq_curves_db[win_idx], -20, 20)
```

### 10. Replace lines 486-498 (apply final EQ):
```python
# OLD:
refined_mag = np.zeros_like(working_mag)
for win_idx in range(num_windows):
    window_mag = working_mag[:, win_idx]
    freq_bins_stft = np.linspace(0, sr/2, len(window_mag))
    freq_points_eq = np.linspace(0, sr/2, 400)
    eq_curve_full = np.interp(freq_bins_stft, freq_points_eq, eq_curves[win_idx])
    refined_mag[:, win_idx] = window_mag * eq_curve_full

# NEW:
refined_mag_db = np.zeros_like(working_mag_db)
for win_idx in range(num_windows):
    window_mag_db = working_mag_db[:, win_idx]
    freq_bins_stft = np.linspace(0, sr/2, len(window_mag_db))
    freq_points_eq = np.linspace(0, sr/2, 400)
    eq_curve_db_full = np.interp(freq_bins_stft, freq_points_eq, eq_curves_db[win_idx])
    refined_mag_db[:, win_idx] = window_mag_db + eq_curve_db_full

# Convert back to linear
refined_mag = librosa.db_to_amplitude(refined_mag_db, ref=1.0)
refined_mag = np.nan_to_num(refined_mag, nan=0.0, posinf=0.0, neginf=0.0)
```

## SUMMARY

**Core transformation:**
- Linear-space (multiplicative): `output = input * eq_curve`
- dB-space (additive): `output_db = input_db + eq_curve_db`

**Key differences:**
1. All magnitudes converted to dB at start
2. EQ curves start at 0 (not 1.0)
3. EQ applied additively (not multiplicatively)
4. Gradient simpler (no mixture_fp multiplication)
5. Clip range is ±20 dB (not 0.1-3.0)
6. Convert back to linear at the end

**Result:**
- Numerical stability from librosa dB conversion
- No NaN propagation (the bug we fixed!)
- Should sound great like Strategy 3 in test_gain_strategies.py
