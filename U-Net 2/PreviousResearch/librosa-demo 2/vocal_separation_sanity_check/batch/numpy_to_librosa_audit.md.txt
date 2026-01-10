# NumPy to Librosa Audit
**Date**: 2025-11-05
**Purpose**: Audit all NumPy usage in vocal separation scripts and identify librosa equivalents

## Summary
This document examines each `np.*` function we use and checks if librosa provides an audio-specific alternative.

---

## NumPy Functions Found

### 1. Array Creation
| NumPy Function | Usage Count | Current Purpose | Librosa Alternative? | Recommendation |
|----------------|-------------|-----------------|----------------------|----------------|
| `np.array()` | 5 | Creating convolution kernels | ❌ No librosa equivalent | **KEEP** - Generic array creation |
| `np.zeros_like()` | 5 | Creating empty spectrograms | ❌ No librosa equivalent | **KEEP** - Generic array op |
| `np.ones()` | 5 | Initializing EQ curves | ❌ No librosa equivalent | **KEEP** - Generic array op |
| `np.zeros()` | 2 | Initializing dB EQ curves | ❌ No librosa equivalent | **KEEP** - Generic array op |

### 2. Mathematical Operations
| NumPy Function | Usage Count | Current Purpose | Librosa Alternative? | Recommendation |
|----------------|-------------|-----------------|----------------------|----------------|
| `np.abs()` | 3 | STFT magnitude extraction | ✅ `librosa.magphase()` | **REPLACE** - Handles zero bins safely |
| `np.angle()` | 1 | Extracting STFT phase | ✅ `librosa.magphase()` | **REPLACE** - Safer phase extraction |
| `np.log10()` | 2 | Converting to dB space | ✅ `librosa.amplitude_to_db()` | **REPLACE** - Audio-aware, numerically stable |
| `np.max()` | 13 | Finding max magnitude | ⚠️ `librosa.util.normalize()` context | **KEEP** - Generic array op |
| `np.mean()` | 6 | Computing loss, averages | ❌ No librosa equivalent | **KEEP** - Generic math |
| `np.exp()` | 1 | Reconstructing complex STFT | ❌ No librosa equivalent | **KEEP** - Standard math |
| `np.clip()` | 5 | Limiting EQ curve range | ❌ No librosa equivalent | **KEEP** - Generic array op |

### 3. Interpolation
| NumPy Function | Usage Count | Current Purpose | Librosa Alternative? | Recommendation |
|----------------|-------------|-----------------|----------------------|----------------|
| `np.interp()` | 15 | Interpolating EQ curves to frequency bins | ⚠️ `scipy.interpolate.interp1d` | **CONSIDER** - scipy is more flexible |
| `np.linspace()` | 16 | Creating frequency bin arrays | ❌ No librosa equivalent | **KEEP** - Generic utility |

### 4. Data Cleaning
| NumPy Function | Usage Count | Current Purpose | Librosa Alternative? | Recommendation |
|----------------|-------------|-----------------|----------------------|----------------|
| `np.nan_to_num()` | 2 | Handling NaN/Inf in fingerprints | ❌ No librosa equivalent | **KEEP** - Generic utility |
| `np.errstate()` | 1 | Suppressing log warnings | ⚠️ Use librosa.amplitude_to_db() | **REPLACE** - Better approach |

---

## Key Findings & Recommendations

### ✅ HIGH PRIORITY - Replace These (Audio-Specific)

1. **`np.abs(stft)` + `np.angle(stft)` → `librosa.magphase(stft)`**
   - **Why**: Handles zero-magnitude bins safely (avoids NaN in phase)
   - **Impact**: More robust STFT processing
   - **Location**: `reconstruct_audio_from_magnitude()`, magnitude extraction
   - **Note**: magphase() returns phase as complex phasor, not angle. Use with `util.phasor()` for reconstruction.

2. **`20 * np.log10(x)` → `librosa.amplitude_to_db(x)`**
   - **Why**: Audio-aware dB conversion with proper reference handling, amin thresholding
   - **Impact**: Prevents log(0) errors, better numerical stability
   - **Location**: Strategy 3 (dB-space EQ)

3. **`magnitude * np.exp(1j * phase)` → `librosa.util.phasor(phase, mag=magnitude)`**
   - **Why**: librosa's specialized function for phase/magnitude to complex conversion
   - **Impact**: Better handling of phase accumulation and edge cases
   - **Location**: `reconstruct_audio_from_magnitude()`
   - **Alternative**: If we have original complex STFT, consider preserving it and scaling magnitude in complex domain

4. **`np.errstate()` context → Remove, use librosa functions**
   - **Why**: librosa functions handle edge cases internally
   - **Impact**: Cleaner code, no warning suppression needed
   - **Location**: `window_to_bottleneck()`

### ⚠️ MEDIUM PRIORITY - Consider These

5. **`np.max(np.abs(audio))` → `librosa.util.normalize(audio)` pattern**
   - **Why**: librosa.util.normalize is audio-aware and handles normalization holistically
   - **Impact**: More consistent with librosa ecosystem
   - **Location**: Final audio normalization in `reconstruct_audio_from_magnitude()`

6. **`np.interp()` → `scipy.interpolate.interp1d()`**
   - **Why**: More flexible interpolation (linear, cubic, etc.)
   - **Impact**: Potentially smoother EQ curve interpolation
   - **Location**: All EQ curve interpolation (15 instances)

### ❌ KEEP These (Generic Operations)

- Array creation: `np.zeros()`, `np.ones()`, `np.array()`, `np.zeros_like()`
- Math: `np.mean()`, `np.exp()`, `np.clip()`
- Utilities: `np.linspace()`, `np.nan_to_num()`

These are generic operations with no audio-specific alternatives.

---

## Implementation Priority

**Phase 1** (Immediate, High Value):
1. Replace `np.abs()` + `np.angle()` with `librosa.magphase()`
2. Replace `magnitude * np.exp(1j * phase)` with `librosa.util.phasor()`
3. Replace `20 * np.log10()` with `librosa.amplitude_to_db()`
4. Remove `np.errstate()` context managers

**Phase 2** (Next, Quality Improvement):
5. Update final normalization to use `librosa.util.normalize()`
6. Test different `axis` parameters for intermediate magnitude normalization

**Phase 3** (Optional, Minor Improvement):
7. Consider `scipy.interpolate.interp1d()` for EQ curve interpolation

---

## Documentation Sources

- librosa.magphase: https://context7.com/websites/librosa_doc/llms.txt?topic=magnitude+phase+stft
- librosa.amplitude_to_db: https://context7.com/websites/librosa_doc/llms.txt?topic=db+amplitude+decibel
- librosa.util.normalize: https://context7.com/websites/librosa_doc/llms.txt?topic=normalize
- librosa.resample/interpolate: https://context7.com/websites/librosa_doc/llms.txt?topic=resample+interpolate
- Complex STFT & reconstruction: https://context7.com/websites/librosa_doc/llms.txt?topic=complex+STFT
- librosa.util.phasor: https://context7.com/websites/librosa_doc/llms.txt?topic=phasor+phase+vocoder

---

## Next Steps

1. ✅ Audit complete
2. ⏳ Create test script with Phase 1 replacements
3. ⏳ Compare audio quality before/after
4. ⏳ If successful, apply to all scripts (test_gain_strategies.py, sanity_check_progressive.py, etc.)
5. ⏳ Update test_normalization_strategies.py with librosa approach

---

## Notes

- **librosa.magphase()** is the single most important replacement - it prevents NaN issues
- **librosa.amplitude_to_db()** should replace all manual log10 conversions for audio
- **librosa.util.phasor()** is the proper way to reconstruct complex STFT from mag+phase
- Most array operations (`np.zeros`, `np.ones`, `np.clip`) have no librosa equivalent and should stay NumPy
- **scipy.ndimage.convolve** is correct for 2D spectral convolutions - librosa has no alternative
- 2D convolution kernels (horizontal, vertical, diagonal filters) are generic image processing operations, not audio-specific
- `np.interp()` works fine for our use case, but `scipy.interpolate.interp1d` offers more flexibility
- The context7.com/websites/librosa_doc/llms.txt endpoint is excellent for querying librosa documentation
