# Librosa Spectrogram Documentation Search Results

**Search URL**: https://context7.com/websites/librosa_doc/llms.txt?topic=spectrogram&tokens=11000

## What We're Currently Using ✅

1. ✅ **`librosa.stft(y)`** - Linear STFT (Short-Time Fourier Transform)
2. ✅ **`librosa.magphase(stft)`** - Magnitude/phase separation
3. ✅ **`librosa.amplitude_to_db()`** - dB conversion for optimization (Strategy 3)
4. ✅ **`librosa.istft()`** - Inverse STFT for reconstruction

## What Librosa Documentation Recommends

### 1. Mel Spectrogram
**`librosa.feature.melspectrogram()`**
- **Purpose**: "Computes a mel-scaled spectrogram from an audio time-series"
- **Why**: Aligns with human auditory perception
- **Benefit**: Better frequency representation for vocals

**Current**: We use **linear STFT** (equal spacing across all frequencies)
**Alternative**: **Mel spectrogram** (perceptually-spaced frequencies)

### 2. Harmonic-Percussive Separation
**`librosa.effects.hpss()`**
- **Purpose**: Separate harmonic (vocals) from percussive (drums) components
- **Why**: Built-in vocal isolation
- **Note**: We're already doing custom separation via gradient descent on spectral slices

### 3. Display Functions
**`librosa.display.specshow()`**
- **Purpose**: Proper spectrogram visualization with axes
- **Current**: We use `ax.imshow()` directly
- **Improvement**: Could add proper frequency/time axis labels

## Key Question: Linear STFT vs Mel Spectrogram?

### Linear STFT (Current Approach)
**Pros:**
- ✅ Full frequency resolution (1025 bins at 22050 Hz)
- ✅ Direct frequency-to-bin mapping
- ✅ Works with our current optimization (400-point EQ curves)
- ✅ **Already tested and sounds great** (Strategy 3 dB-space)

**Cons:**
- ⚠️ Not perceptually motivated
- ⚠️ More bins at high frequencies where humans are less sensitive

### Mel Spectrogram (Librosa Recommendation)
**Pros:**
- ✅ Perceptually motivated (logarithmic frequency scale)
- ✅ More bins at low frequencies (where vocals are)
- ✅ Standard in speech/music processing

**Cons:**
- ❌ Would require **complete pipeline rewrite**
- ❌ Different dimensionality (typically 128 mel bins, not 1025 linear bins)
- ❌ Different interpolation math (mel scale → frequency scale)
- ❌ **Would invalidate all current testing**

## Recommendation

**KEEP LINEAR STFT** because:

1. ✅ **Working system**: Strategy 3 (dB-space) sounds great with current approach
2. ✅ **Full frequency resolution**: 1025 bins gives detailed control
3. ✅ **Tested**: User verified "sounds Greta!" with current implementation
4. ✅ **Simpler**: Direct frequency mapping, no scale conversion needed

**Mel spectrograms are great for:**
- Machine learning models (CNNs, Transformers)
- Speech recognition
- Music classification

**But our approach uses:**
- Gradient descent on raw frequency profiles
- Hand-crafted spectral slice extraction
- Direct EQ curve optimization

**Linear STFT is better suited for our gradient-based optimization approach.**

## What We Should Actually Do

### Immediate (Keep Current):
1. ✅ Continue using linear STFT
2. ✅ Keep Strategy 3 dB-space optimization
3. ✅ Keep all 18 spectral slices as-is

### Optional Enhancement (Later):
1. ⚠️ Consider using `librosa.display.specshow()` for visualization (better axes)
2. ⚠️ Could experiment with mel spectrograms in a **separate branch** to compare
3. ⚠️ Document that linear STFT is intentional choice for gradient-based approach

## Conclusion

**Don't switch to mel spectrograms.**

Our linear STFT approach is:
- Working well (Strategy 3 sounds great)
- Better suited for gradient descent optimization
- Provides full frequency resolution
- Already tested and proven

Mel spectrograms are recommended for **feature-based ML**, not **gradient-based signal processing**.
