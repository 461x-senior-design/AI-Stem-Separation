# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    start_time = time.time()

    print("\n" + "="*70)
    print("PHASE 1: LOAD AND ANALYZE")
    print("="*70)

    # Process vocal
    vocal_audio, vocal_stft, vocal_mag, vocal_fps = process_audio_to_fingerprints(
        CONFIG['vocal_path'],
        CONFIG['sr'],
        CONFIG['n_fft'],
        CONFIG['hop_length']
    )

    # Process mixture
    mixture_audio, mixture_stft, mixture_mag, mixture_fps = process_audio_to_fingerprints(
        CONFIG['mixture_path'],
        CONFIG['sr'],
        CONFIG['n_fft'],
        CONFIG['hop_length']
    )

    num_windows = mixture_mag.shape[1]
    # Extract phase as complex phasor using librosa.magphase
    _, mixture_phase = librosa.magphase(mixture_stft)

    # Slice order for testing (1-5)
    slice_names = [
        'slice_1_horizontal',
        'slice_2_vertical',
        'slice_3_diagonal_up',
        'slice_4_diagonal_down',
        'slice_5_blob',
    ]

    print(f"\n✓ Fingerprints created")

    # ============================================
    # PHASE 2: TEST ALL 4 STRATEGIES
    # ============================================

    print("\n" + "="*70)
    print("PHASE 2: TEST ALL 4 STRATEGIES")
    print("="*70)
    print(f"\nUsing pipeline: {' → '.join(slice_names)} → slice_0_raw")
    print()

    results = {}

    # Save reference files once at the beginning
    sf.write(Path(CONFIG['output_dir']) / "0_original_mixture.wav", mixture_audio, CONFIG['sr'])
    sf.write(Path(CONFIG['output_dir']) / "0_target_vocal.wav", vocal_audio, CONFIG['sr'])
    print(f"\n✓ Saved reference files in {CONFIG['output_dir']}/")

    # ============================================
    # TEST 1: NORMALIZE AFTER EACH PASS
    # ============================================

    print("\n" + "="*70)
    print("STRATEGY 1/4: NORMALIZE AFTER EACH PASS")
    print("="*70)

    current_mag = None
    for i, slice_name in enumerate(slice_names, 1):
        print(f"\nPass {i}/{len(slice_names)}: {slice_name}")
        current_mag = optimize_strategy1_normalize(
            slice_name, vocal_fps, mixture_mag, num_windows, CONFIG['sr'], current_mag
        )
        print(f"  ✓ Max magnitude: {np.max(current_mag):.2f}")

    # Final polish with slice_0
    print(f"\nFinal polish: slice_0_raw")
    final_mag = optimize_strategy1_normalize(
        'slice_0_raw', vocal_fps, mixture_mag, num_windows, CONFIG['sr'], current_mag
    )
    print(f"  ✓ Max magnitude: {np.max(final_mag):.2f}")

    audio1 = reconstruct_audio_from_magnitude(final_mag, mixture_phase, CONFIG['hop_length'], CONFIG['n_fft'])
    results['strategy1_normalize'] = audio1

    # SAVE IMMEDIATELY
    output_path1 = Path(CONFIG['output_dir']) / "strategy1_normalize.wav"
    sf.write(output_path1, audio1, CONFIG['sr'])
    print(f"\n✓ Strategy 1 complete!")
    print(f"  📁 Saved: {output_path1}")
    print(f"  🎧 Ready to listen!")

    # ============================================
    # TEST 2: TRACK GAIN AND COMPENSATE
    # ============================================

    print("\n" + "="*70)
    print("STRATEGY 2/4: TRACK GAIN AND COMPENSATE")
    print("="*70)

    current_mag = None
    accumulated_gains = None

    for i, slice_name in enumerate(slice_names, 1):
        print(f"\nPass {i}/{len(slice_names)}: {slice_name}")
        current_mag, accumulated_gains = optimize_strategy2_compensate(
            slice_name, vocal_fps, mixture_mag, num_windows, CONFIG['sr'], current_mag, accumulated_gains
        )
        print(f"  ✓ Max magnitude: {np.max(current_mag):.2f}")

    # Final polish
    print(f"\nFinal polish: slice_0_raw")
    final_mag, accumulated_gains = optimize_strategy2_compensate(
        'slice_0_raw', vocal_fps, mixture_mag, num_windows, CONFIG['sr'], current_mag, accumulated_gains
    )
    print(f"  ✓ Max magnitude: {np.max(final_mag):.2f}")

    # Apply inverse gain compensation
    print("\nApplying gain compensation...")
    compensated_mag = np.zeros_like(final_mag)
    for win_idx in range(num_windows):
        window_mag = final_mag[:, win_idx]

        # Calculate average accumulated gain
        avg_gain = np.mean(accumulated_gains[win_idx])

        # Apply inverse
        if avg_gain > 0:
            compensation_factor = 1.0 / avg_gain
            freq_bins_stft = np.linspace(0, CONFIG['sr']/2, len(window_mag))
            freq_points_eq = np.linspace(0, CONFIG['sr']/2, 400)
            comp_curve = np.ones(400) * compensation_factor
            comp_curve_full = np.interp(freq_bins_stft, freq_points_eq, comp_curve)
            compensated_mag[:, win_idx] = window_mag * comp_curve_full
        else:
            compensated_mag[:, win_idx] = window_mag

    print(f"  ✓ Compensated max magnitude: {np.max(compensated_mag):.2f}")

    audio2 = reconstruct_audio_from_magnitude(compensated_mag, mixture_phase, CONFIG['hop_length'], CONFIG['n_fft'])
    results['strategy2_compensate'] = audio2

    # SAVE IMMEDIATELY
    output_path2 = Path(CONFIG['output_dir']) / "strategy2_compensate.wav"
    sf.write(output_path2, audio2, CONFIG['sr'])
    print(f"\n✓ Strategy 2 complete!")
    print(f"  📁 Saved: {output_path2}")
    print(f"  🎧 Ready to listen!")

    # ============================================
    # TEST 3: DB-SPACE EQ
    # ============================================

    print("\n" + "="*70)
    print("STRATEGY 3/4: DB-SPACE EQ")
    print("="*70)

    current_mag = None
    for i, slice_name in enumerate(slice_names, 1):
        print(f"\nPass {i}/{len(slice_names)}: {slice_name}")
        current_mag = optimize_strategy3_db_space(
            slice_name, vocal_fps, mixture_mag, num_windows, CONFIG['sr'], current_mag
        )
        print(f"  ✓ Max magnitude: {np.max(current_mag):.2f}")

    # Final polish
    print(f"\nFinal polish: slice_0_raw")
    final_mag = optimize_strategy3_db_space(
        'slice_0_raw', vocal_fps, mixture_mag, num_windows, CONFIG['sr'], current_mag
    )
    print(f"  ✓ Max magnitude: {np.max(final_mag):.2f}")

    audio3 = reconstruct_audio_from_magnitude(final_mag, mixture_phase, CONFIG['hop_length'], CONFIG['n_fft'])
    results['strategy3_dbspace'] = audio3

    # SAVE IMMEDIATELY
    output_path3 = Path(CONFIG['output_dir']) / "strategy3_dbspace.wav"
    sf.write(output_path3, audio3, CONFIG['sr'])
    print(f"\n✓ Strategy 3 complete!")
    print(f"  📁 Saved: {output_path3}")
    print(f"  🎧 Ready to listen!")

    # ============================================
    # TEST 4: SYMMETRIC CLIPPING
    # ============================================

    print("\n" + "="*70)
    print("STRATEGY 4/4: SYMMETRIC CLIPPING RANGE")
    print("="*70)

    current_mag = None
    for i, slice_name in enumerate(slice_names, 1):
        print(f"\nPass {i}/{len(slice_names)}: {slice_name}")
        current_mag = optimize_strategy4_symmetric(
            slice_name, vocal_fps, mixture_mag, num_windows, CONFIG['sr'], current_mag
        )
        print(f"  ✓ Max magnitude: {np.max(current_mag):.2f}")

    # Final polish
    print(f"\nFinal polish: slice_0_raw")
    final_mag = optimize_strategy4_symmetric(
        'slice_0_raw', vocal_fps, mixture_mag, num_windows, CONFIG['sr'], current_mag
    )
    print(f"  ✓ Max magnitude: {np.max(final_mag):.2f}")

    audio4 = reconstruct_audio_from_magnitude(final_mag, mixture_phase, CONFIG['hop_length'], CONFIG['n_fft'])
    results['strategy4_symmetric'] = audio4

    # SAVE IMMEDIATELY
    output_path4 = Path(CONFIG['output_dir']) / "strategy4_symmetric.wav"
    sf.write(output_path4, audio4, CONFIG['sr'])
    print(f"\n✓ Strategy 4 complete!")
    print(f"  📁 Saved: {output_path4}")
    print(f"  🎧 Ready to listen!")

    # ============================================
    # PHASE 3: VISUALIZATION
    # ============================================

    print("\n" + "="*70)
    print("PHASE 3: CREATING COMPARISON VISUALIZATION")
    print("="*70)

    print("\nGenerating comparison spectrograms...")

    fig, axes = plt.subplots(6, 1, figsize=(14, 18))

    # Target vocal
    axes[0].imshow(librosa.amplitude_to_db(vocal_mag, ref=np.max),
                   aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Target Vocal (Reference)')
    axes[0].set_ylabel('Frequency')

    # Original mixture
    axes[1].imshow(librosa.amplitude_to_db(mixture_mag, ref=np.max),
                   aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title('Original Mixture')
    axes[1].set_ylabel('Frequency')

    # All 4 strategies
    strategy_titles = [
        'Strategy 1: Normalize After Each Pass',
        'Strategy 2: Track Gain & Compensate',
        'Strategy 3: dB-Space EQ',
        'Strategy 4: Symmetric Clipping [0.333, 3.0]'
    ]

    for idx, (strategy_name, title) in enumerate(zip(results.keys(), strategy_titles)):
        audio = results[strategy_name]
        stft = librosa.stft(audio, n_fft=CONFIG['n_fft'], hop_length=CONFIG['hop_length'])
        mag = np.abs(stft)

        axes[idx + 2].imshow(librosa.amplitude_to_db(mag, ref=np.max),
                            aspect='auto', origin='lower', cmap='viridis')
        axes[idx + 2].set_title(title)
        axes[idx + 2].set_ylabel('Frequency')

    axes[-1].set_xlabel('Time')

    plt.tight_layout()
    viz_path = Path(CONFIG['output_dir']) / "comparison_spectrograms.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {viz_path}")

    # ============================================
    # FINAL SUMMARY
    # ============================================

    total_time = time.time() - start_time

    print("\n" + "="*70)
    print("✓ GAIN STRATEGY TEST COMPLETE!")
    print("="*70)

    print(f"\nTotal runtime: {total_time:.1f}s")
    print(f"\nOutput directory: {CONFIG['output_dir']}/")
    print(f"\nGenerated files:")
    print(f"  • strategy1_normalize.wav       ← Normalize after each pass")
    print(f"  • strategy2_compensate.wav      ← Track gain & compensate")
    print(f"  • strategy3_dbspace.wav         ← dB-space EQ")
    print(f"  • strategy4_symmetric.wav       ← Symmetric clipping")
    print(f"  • 0_original_mixture.wav        ← Reference")
    print(f"  • 0_target_vocal.wav            ← Reference")
    print(f"  • comparison_spectrograms.png   ← Visual comparison")

    print("\n🎧 Listen to all 4 outputs and compare:")
    print("   - Which one sounds loudest/clearest?")
    print("   - Which one best preserves vocal quality?")
    print("   - Which one has least artifacts?")

    print("\n" + "="*70)
