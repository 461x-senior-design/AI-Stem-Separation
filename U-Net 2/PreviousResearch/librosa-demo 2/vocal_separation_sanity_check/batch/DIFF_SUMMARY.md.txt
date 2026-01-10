# Diff Summary: test_gain_strategies.py vs sanity_check_progressive.py

## ✅ EXACTLY MATCHING (Core Strategy 3 Logic)

### 1. STFT Creation
```python
stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
magnitude, phase = librosa.magphase(stft)
```
✅ IDENTICAL in both files

### 2. Phase Extraction
```python
_, mixture_phase = librosa.magphase(mixture_stft)
```
✅ IDENTICAL in both files (line 426 vs line 560)

### 3. Audio Reconstruction
```python
def reconstruct_audio_from_magnitude(magnitude, phase, hop_length, n_fft):
    reconstructed_stft = magnitude * phase
    audio = librosa.istft(reconstructed_stft, hop_length=hop_length, n_fft=n_fft)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    return audio
```
✅ IDENTICAL in both files

### 4. Core dB-Space Optimization Logic

**Working magnitude setup:**
```python
working_mag_db = librosa.amplitude_to_db(working_mag, ref=1.0, amin=1e-8)
eq_curves_db = [np.zeros(400) for _ in range(num_windows)]
```
✅ IDENTICAL

**Optimization loop - target conversion:**
```python
vocal_fp_db = librosa.amplitude_to_db(vocal_fp, ref=1.0, amin=1e-8)
```
✅ IDENTICAL

**Optimization loop - mixture extraction:**
```python
mixture_window_db = working_mag_db[:, win_idx]
mixture_fp_db = np.interp(
    x=np.linspace(0, sr/2, 400),
    xp=np.linspace(0, sr/2, len(mixture_window_db)),
    fp=mixture_window_db
)
```
✅ IDENTICAL

**Optimization loop - EQ application:**
```python
adjusted_fp_db = mixture_fp_db + eq_curves_db[win_idx]
```
✅ IDENTICAL

**Optimization loop - loss:**
```python
loss = np.mean((adjusted_fp_db - vocal_fp_db)**2)
```
✅ IDENTICAL

**Optimization loop - gradient:**
```python
gradient = 2 * (adjusted_fp_db - vocal_fp_db)
```
✅ IDENTICAL

**Optimization loop - update:**
```python
eq_curves_db[win_idx] -= CONFIG['learning_rate'] * gradient
```
✅ IDENTICAL

**Optimization loop - clipping:**
```python
eq_curves_db[win_idx] = np.clip(eq_curves_db[win_idx], -20, 20)
```
✅ IDENTICAL

**Apply EQ:**
```python
refined_mag_db = np.zeros_like(working_mag_db)
for win_idx in range(num_windows):
    window_mag_db = working_mag_db[:, win_idx]
    freq_bins_stft = np.linspace(0, sr/2, len(window_mag_db))
    freq_points_eq = np.linspace(0, sr/2, 400)
    eq_curve_db_full = np.interp(freq_bins_stft, freq_points_eq, eq_curves_db[win_idx])
    refined_mag_db[:, win_idx] = window_mag_db + eq_curve_db_full
```
✅ IDENTICAL

**Convert back:**
```python
refined_mag = librosa.db_to_amplitude(refined_mag_db, ref=1.0)
refined_mag = np.nan_to_num(refined_mag, nan=0.0, posinf=0.0, neginf=0.0)
```
✅ IDENTICAL

---

## ⚠️ DIFFERENT (Extra Features for Progressive Testing)

### optimize_strategy3_db_space vs optimize_one_slice

**test_gain_strategies.py (optimize_strategy3_db_space):**
- NO print statements
- NO loss tracking
- Returns: `return refined_mag`
- Signature: `(slice_name, vocal_fps, mixture_mag, num_windows, sr, current_mag=None)`

**sanity_check_progressive.py (optimize_one_slice):**
- HAS print statements for progress tracking:
  ```python
  print(f"\n{'='*70}")
  print(f"OPTIMIZING: {slice_name}")
  print(f"  Starting from: ...")
  print(f"  Iteration {display_iter:3d}: Loss = {avg_loss:.6f}")
  print(f"  ✓ Optimization complete!")
  ```
- HAS loss tracking:
  ```python
  losses = []
  avg_loss = total_loss / num_windows
  losses.append(avg_loss)
  ```
- Returns: `return refined_mag, losses`
- Signature: `(slice_name, vocal_fps, mixture_mag, num_windows, sr, current_mag=None, iteration_offset=0)`
- Has `iteration_offset` parameter for displaying iteration numbers across multiple passes

---

## 🎯 CONCLUSION

**THE CORE STRATEGY 3 dB-SPACE OPTIMIZATION LOGIC IS 100% IDENTICAL! ✅**

The differences are ONLY:
1. **Logging** (print statements for progress)
2. **Tracking** (losses list for visualization)
3. **Return value** (includes losses for plotting)
4. **iteration_offset** (for multi-pass iteration display)

**These differences do NOT affect the audio processing!** They're just for:
- User feedback during long runs
- Loss visualization in plots
- Multi-slice iteration counting

**The actual math that produces the audio is EXACTLY THE SAME as Strategy 3!** ✅
