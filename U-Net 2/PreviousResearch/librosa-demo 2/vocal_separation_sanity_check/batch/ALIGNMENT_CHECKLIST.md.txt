# COMPLETE ALIGNMENT CHECKLIST

## Functions that MUST match test_gain_strategies.py Strategy 3:

### ✅ 1. window_to_bottleneck()
- [x] Direct interpolation to 400 points (no encoder)
- [x] Returns only freq_profile_400

### ✅ 2. process_audio_to_fingerprints() - STFT creation
- [x] librosa.stft()
- [x] librosa.magphase() for magnitude extraction

### ✅ 3. Phase extraction (main execution)
- [x] _, mixture_phase = librosa.magphase(mixture_stft)

### ✅ 4. optimize_one_slice() / optimize_strategy3_db_space()
- [x] librosa.amplitude_to_db(working_mag, ref=1.0, amin=1e-8)
- [x] eq_curves_db = [np.zeros(400)]
- [x] vocal_fp_db = librosa.amplitude_to_db(vocal_fp, ref=1.0, amin=1e-8)
- [x] adjusted_fp_db = mixture_fp_db + eq_curves_db[win_idx]
- [x] gradient = 2 * (adjusted_fp_db - vocal_fp_db)
- [x] np.clip(eq_curves_db[win_idx], -20, 20)
- [x] librosa.db_to_amplitude(refined_mag_db, ref=1.0)
- [x] np.nan_to_num()

### ✅ 5. reconstruct_audio_from_magnitude()
- [x] reconstructed_stft = magnitude * phase
- [x] librosa.istft()
- [x] audio / (np.max(np.abs(audio)) + 1e-8)

### ✅ 6. Visualization - librosa.magphase() for spectrograms
- [x] Uses librosa.magphase() not np.abs()

## ALL CRITICAL FUNCTIONS NOW MATCH! ✅
