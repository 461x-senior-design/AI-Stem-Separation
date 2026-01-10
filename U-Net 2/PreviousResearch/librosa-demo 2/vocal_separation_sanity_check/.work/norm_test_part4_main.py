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
    mixture_phase = np.angle(mixture_stft)

    # Slice order for testing (1-5)
    slice_names = [
        'slice_1_horizontal',
        'slice_2_vertical',
        'slice_3_diagonal_up',
        'slice_4_diagonal_down',
        'slice_5_blob',
    ]

    print(f"\n✓ Fingerprints created")

    # Save reference files once at the beginning
    sf.write(Path(CONFIG['output_dir']) / "0_original_mixture.wav", mixture_audio, CONFIG['sr'])
    sf.write(Path(CONFIG['output_dir']) / "0_target_vocal.wav", vocal_audio, CONFIG['sr'])
    print(f"\n✓ Saved reference files in {CONFIG['output_dir']}/")

    # ============================================
    # PHASE 2: TEST ALL 4 NORMALIZATION STRATEGIES
    # ============================================

    print("\n" + "="*70)
    print("PHASE 2: TEST ALL 4 NORMALIZATION STRATEGIES")
    print("="*70)
    print(f"\nUsing pipeline: {' → '.join(slice_names)} → slice_0_raw")
    print()

    results = {}
    strategies = [
        ('manual', 'Manual Global Normalization'),
        ('librosa_global', 'librosa.util.normalize (global)'),
        ('librosa_axis0', 'librosa.util.normalize (axis=0, per-freq)'),
        ('librosa_axis1', 'librosa.util.normalize (axis=1, per-time)'),
    ]

    for strategy_idx, (strategy_key, strategy_name) in enumerate(strategies, 1):
        print("\n" + "="*70)
        print(f"STRATEGY {strategy_idx}/4: {strategy_name}")
        print("="*70)

        current_mag = None

        # Process through 5 slices
        for i, slice_name in enumerate(slice_names, 1):
            print(f"\nPass {i}/{len(slice_names)}: {slice_name}")
            current_mag = optimize_with_normalization(
                slice_name, vocal_fps, mixture_mag, num_windows,
                CONFIG['sr'], current_mag, strategy_key
            )
            print(f"  ✓ Max magnitude: {np.max(current_mag):.2f}")

        # Final polish with slice_0
        print(f"\nFinal polish: slice_0_raw")
        final_mag = optimize_with_normalization(
            'slice_0_raw', vocal_fps, mixture_mag, num_windows,
            CONFIG['sr'], current_mag, strategy_key
        )
        print(f"  ✓ Max magnitude: {np.max(final_mag):.2f}")

        # Reconstruct audio
        audio = reconstruct_audio_from_magnitude(
            final_mag, mixture_phase, CONFIG['hop_length'], CONFIG['n_fft']
        )
        results[strategy_key] = audio

        # Save immediately
        output_path = Path(CONFIG['output_dir']) / f"{strategy_key}.wav"
        sf.write(output_path, audio, CONFIG['sr'])
        print(f"\n✓ Strategy {strategy_idx} complete!")
        print(f"  📁 Saved: {output_path}")
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
    strategy_titles = [name for _, name in strategies]

    for idx, ((strategy_key, _), title) in enumerate(zip(strategies, strategy_titles)):
        audio = results[strategy_key]
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
    print("✓ NORMALIZATION STRATEGY TEST COMPLETE!")
    print("="*70)

    print(f"\nTotal runtime: {total_time:.1f}s")
    print(f"\nOutput directory: {CONFIG['output_dir']}/")
    print(f"\nGenerated files:")
    print(f"  • manual.wav               ← Manual global normalization (current)")
    print(f"  • librosa_global.wav       ← librosa.util.normalize (global)")
    print(f"  • librosa_axis0.wav        ← librosa.util.normalize (per-frequency)")
    print(f"  • librosa_axis1.wav        ← librosa.util.normalize (per-time)")
    print(f"  • 0_original_mixture.wav   ← Reference")
    print(f"  • 0_target_vocal.wav       ← Reference")
    print(f"  • comparison_spectrograms.png  ← Visual comparison")

    print("\n🎧 Listen to all 4 outputs and compare:")
    print("   - Which one sounds clearest?")
    print("   - Which has best vocal isolation?")
    print("   - Which has least artifacts?")

    print("\n" + "="*70)
