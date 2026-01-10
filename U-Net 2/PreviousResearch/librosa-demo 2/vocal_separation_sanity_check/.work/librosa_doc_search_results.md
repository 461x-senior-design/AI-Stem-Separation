# Librosa Documentation Search Results - Magnitude Operations

**Search URL**: https://context7.com/websites/librosa_doc/llms.txt?topic=magnitude&tokens=11000

## What We're Already Using ✅

1. ✅ **`librosa.stft()`** - STFT computation
2. ✅ **`librosa.magphase(D)`** - Magnitude/phase separation (replaced np.abs + np.angle)
3. ✅ **`librosa.amplitude_to_db()`** - dB conversion (replaced 20 * np.log10)
4. ✅ **`librosa.istft()`** - Inverse STFT
5. ✅ **`magnitude * phase`** - Reconstruction (phase is phasor from magphase)

## New Functions Found

### librosa.util.abs2(x)
- **Purpose**: Computes squared magnitude more efficiently than `np.abs(x)**2`
- **Our Usage**: We only square differences for loss: `(adjusted_fp_db - vocal_fp_db)**2`
- **Recommendation**: ❌ NOT APPLICABLE - We're not squaring magnitudes

### librosa.griffinlim(S)
- **Purpose**: Phase reconstruction from magnitude-only spectrograms
- **Our Usage**: We already have phase from original STFT
- **Recommendation**: ❌ NOT NEEDED - We preserve phase throughout

### Feature Extraction Functions
- `librosa.feature.melspectrogram()` - Mel-scale mapping
- `librosa.feature.chroma_stft()` - Chromatic features
- `librosa.salience()` - Harmonic salience
- `librosa.feature.rms()` - Frame-wise energy

**Recommendation**: ❌ NOT APPLICABLE - These are for feature-based approaches. We use gradient descent on raw frequency profiles, not features.

## Conclusion

✅ **WE ARE ALREADY USING ALL APPLICABLE LIBROSA FUNCTIONS!**

Our current implementation:
- Uses librosa for ALL audio-specific operations
- Uses numpy only for generic array/math operations
- Matches best practices from librosa documentation

**No additional librosa functions needed!** 🎯

## References
- librosa.magphase: "Separates a complex-valued spectrogram into its magnitude and phase components"
- librosa.util.phasor: For constructing complex phasors (our phase is already phasor from magphase)
- librosa.griffinlim: For magnitude-only reconstruction (we have phase, so not needed)
