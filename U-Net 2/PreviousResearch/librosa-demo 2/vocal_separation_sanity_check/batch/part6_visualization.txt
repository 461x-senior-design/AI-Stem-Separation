    # ============================================
    # PHASE 3: VISUALIZATION AND SUMMARY
    # ============================================

    print("\n" + "="*70)
    print("PHASE 3: VISUALIZATION AND SUMMARY")
    print("="*70)

    # Create comparison plots
    print("\nGenerating visualizations...")

    # Plot 1: Loss progression for all tests
    fig, ax = plt.subplots(figsize=(14, 6))

    for result in all_results:
        ax.plot(result['losses'], label=f"{result['num_slices']} slice(s)", alpha=0.7)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Optimization Loss Progression (All Tests)')
    ax.legend(loc='upper right', ncol=3)
    ax.grid(True, alpha=0.3)

    loss_plot_path = Path(CONFIG['output_base_dir']) / "all_tests_loss_progression.png"
    plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {loss_plot_path}")

    # Plot 2: Final loss comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    num_slices_list = [r['num_slices'] for r in all_results]
    final_losses = [r['final_loss'] for r in all_results]

    ax.bar(num_slices_list, final_losses, color='steelblue', alpha=0.7)
    ax.set_xlabel('Number of Conv Slices Used')
    ax.set_ylabel('Final Loss (MSE)')
    ax.set_title('Final Loss vs Number of Slices')
    ax.set_xticks(num_slices_list)
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate best result
    best_idx = np.argmin(final_losses)
    ax.axhline(y=final_losses[best_idx], color='red', linestyle='--', alpha=0.5, label='Best')
    ax.legend()

    loss_comparison_path = Path(CONFIG['output_base_dir']) / "final_loss_comparison.png"
    plt.savefig(loss_comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {loss_comparison_path}")

    # Plot 3: Sample spectrograms (first, middle, last tests)
    test_indices = [0, len(all_results) // 2, len(all_results) - 1] if len(all_results) >= 3 else range(len(all_results))

    fig, axes = plt.subplots(len(test_indices) + 2, 1, figsize=(14, 4 * (len(test_indices) + 2)))

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

    # Sample extracted results
    for plot_idx, result_idx in enumerate(test_indices):
        result = all_results[result_idx]
        audio_path = result['output_path']

        # Load and compute spectrogram
        audio, _ = librosa.load(audio_path, sr=CONFIG['sr'])
        stft = librosa.stft(audio, n_fft=CONFIG['n_fft'], hop_length=CONFIG['hop_length'])
        mag = np.abs(stft)

        ax = axes[plot_idx + 2]
        ax.imshow(librosa.amplitude_to_db(mag, ref=np.max),
                  aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(f"Extracted ({result['num_slices']} slice(s), loss={result['final_loss']:.4f})")
        ax.set_ylabel('Frequency')

        if plot_idx == len(test_indices) - 1:
            ax.set_xlabel('Time')

    plt.tight_layout()
    spectrogram_comparison_path = Path(CONFIG['output_base_dir']) / "spectrogram_comparison.png"
    plt.savefig(spectrogram_comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {spectrogram_comparison_path}")

    # ============================================
    # FINAL SUMMARY
    # ============================================

    total_time = time.time() - start_time

    print("\n" + "="*70)
    print("✓ PROGRESSIVE TEST COMPLETE!")
    print("="*70)

    print(f"\nTotal runtime: {total_time:.1f}s")
    print(f"Tests completed: {len(all_results)}")
    print(f"\nResults summary:")

    for result in all_results:
        print(f"  {result['num_slices']:2d} slice(s): loss={result['final_loss']:.6f}, time={result['duration']:.1f}s")

    # Identify best result
    best_result = all_results[best_idx]
    print(f"\n🎵 Best result: {best_result['num_slices']} slice(s)")
    print(f"   Loss: {best_result['final_loss']:.6f}")
    print(f"   Path: {best_result['output_path']}")

    print(f"\n📁 All outputs saved in: {CONFIG['output_base_dir']}/")
    print(f"\n📊 Visualizations:")
    print(f"  • {loss_plot_path.name}")
    print(f"  • {loss_comparison_path.name}")
    print(f"  • {spectrogram_comparison_path.name}")

    print(f"\n🎧 Listen to the outputs to hear progressive improvement!")
    print(f"\nEach test directory contains:")
    print(f"  • extracted_vocal.wav  ← Separated vocal")
    print(f"  • 1_original_mixture.wav (in 1_slices/ only)")
    print(f"  • 2_target_vocal.wav (in 1_slices/ only)")

    print("\n" + "="*70)
