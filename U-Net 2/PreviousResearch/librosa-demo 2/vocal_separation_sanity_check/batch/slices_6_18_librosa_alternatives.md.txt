# Librosa Alternatives for Slices 6-18

## Current Implementation (Manual 2D Convolution Kernels)

**Slices 6-18 use scipy.ndimage.convolve with hand-crafted kernels to create different "views" of the spectrogram**

### Slice 6: Harmonic Stack
```python
kernel_harmonic = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]])
```
- **Purpose**: Detect harmonic stacking patterns (frequencies above/below)
- **Librosa Alternative**: `librosa.effects.harmonic()` or `librosa.salience()`

### Slices 7-8: High-pass / Low-pass
```python
kernel_hp = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])  # high-pass
kernel_lp = np.ones((3, 3)) / 9  # low-pass
```
- **Purpose**: Spectral filtering (emphasize/smooth different frequency ranges)
- **Librosa Alternative**: `librosa.iirt()` or perceptual weighting functions

### Slices 9-15: Oriented Edge Detectors
```python
angles = [22.5, 45, 67.5, 90, 112.5, 135, 157.5]
create_oriented_filter(angle)  # edge detection at various angles
```
- **Purpose**: Detect pitch contours, onsets at different angles
- **Librosa Alternative**: `librosa.onset.onset_strength()` or pitch tracking

### Slice 16: Laplacian
```python
kernel_laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
```
- **Purpose**: Transient/edge detection (all directions)
- **Librosa Alternative**: `librosa.onset.onset_detect()` or `onset_strength()`

### Slices 17-18: MaxPool / AvgPool
```python
ndimage.maximum_filter(magnitude, size=(2, 2))[::2, ::2]  # maxpool
ndimage.uniform_filter(magnitude, size=(2, 2))[::2, ::2]  # avgpool
```
- **Purpose**: Downsampling with max/average aggregation
- **Librosa Alternative**: `librosa.resample()` or decimation?

## Key Question: Should We Replace Them?

### Arguments FOR Replacement:
1. ✅ **Audio-aware**: Librosa functions understand audio semantics
2. ✅ **Tested**: These are well-tested functions used in production
3. ✅ **Simpler**: Less manual kernel design

### Arguments AGAINST Replacement:
1. ⚠️ **Different outputs**: Librosa functions return different data structures
   - `harmonic()` returns time-domain audio, not spectrogram
   - `onset_strength()` returns 1D envelope, not 2D spectrogram
   - Our kernels create 2D "feature maps" for optimization

2. ⚠️ **Integration complexity**: Would need to restructure the entire optimization pipeline
   - Current: Create 18 spectrogram views → Extract 400-point profiles → Optimize
   - New: Would need to figure out how to extract features from librosa outputs

3. ⚠️ **Loss of flexibility**: Hand-crafted kernels let us create specific "views"
   - We control exactly what features to extract
   - Librosa functions have fixed behavior

4. ⚠️ **Testing burden**: We already tested with manual kernels
   - User heard progressive improvement with current approach
   - Switching would require re-testing everything

## Recommendation

**KEEP MANUAL KERNELS FOR NOW** because:

1. **Current approach works**: Creates 18 different spectrogram views
2. **Integration is clean**: All slices have same shape, same fingerprint extraction
3. **Proven in testing**: User tested Strategy 3 (dB-space) and it sounds great

**Consider librosa functions LATER** if:
- We want to explore different feature extraction approaches
- We redesign the optimization pipeline
- We need to compare manual vs. librosa-based features

## What Librosa IS Good For (Already Using)

✅ **STFT/magnitude/phase**: `librosa.stft()`, `librosa.magphase()`
✅ **dB conversion**: `librosa.amplitude_to_db()`, `librosa.db_to_amplitude()`
✅ **Audio reconstruction**: `librosa.istft()`
✅ **Manual normalization**: Works best per user testing

## Conclusion

**Don't replace slices 6-18 with librosa functions.**

The manual 2D convolution kernels are:
- Creating the right kind of outputs (2D spectrograms)
- Integrated cleanly with the optimization pipeline
- Working well in testing

Librosa functions are audio-specific but would require major refactoring to integrate.
