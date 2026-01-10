# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    start_time = time.time()

    # ============================================
    # PHASE 1: LOAD AND ANALYZE
    # ============================================

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

    print(f"\n✓ Fingerprints created in {time.time() - start_time:.1f}s")

    # Get ordered slice names
    all_slice_names = get_slice_order()

    # ============================================
    # PHASE 2: PROGRESSIVE TESTING
    # ============================================

    print("\n" + "="*70)
    print("PHASE 2: PROGRESSIVE TESTING")
    print("="*70)
    print(f"\nTesting {CONFIG['max_slices']} configurations:")
    print("  Each test uses slices 1 through N, then slice_0_raw as final polish")
    print()

    # Store results for comparison
    all_results = []

    # Progressive loop: test 1 slice, 2 slices, 3 slices, ..., N slices
    for num_slices in range(1, CONFIG['max_slices'] + 1):
        test_start_time = time.time()

        print("\n" + "="*70)
        print(f"TEST {num_slices}/{CONFIG['max_slices']}: Using {num_slices} conv slice(s) + slice_0_raw")
        print("="*70)

        # Create output directory for this test
        test_output_dir = Path(CONFIG['output_base_dir']) / f"{num_slices}_slices"
        test_output_dir.mkdir(exist_ok=True, parents=True)

        # Select slices to use (1 through num_slices)
        slices_to_use = all_slice_names[:num_slices]

        print(f"\nPipeline:")
        for i, slice_name in enumerate(slices_to_use, 1):
            print(f"  Pass {i}: {slice_name}")
        print(f"  Pass {len(slices_to_use) + 1}: slice_0_raw (final polish)")
        print()

        # Sequential optimization through selected slices
        current_mag = None
        all_losses = []
        iteration_counter = 0

        for slice_idx, slice_name in enumerate(slices_to_use):
            refined_mag, losses = optimize_one_slice(
                slice_name,
                vocal_fps,
                mixture_mag,
                num_windows,
                CONFIG['sr'],
                current_mag=current_mag,
                iteration_offset=iteration_counter
            )
            current_mag = refined_mag
            all_losses.extend(losses)
            iteration_counter += CONFIG['num_iterations']

        # Final polish with slice_0_raw
        print(f"\n{'='*70}")
        print("FINAL POLISH: slice_0_raw")
        print(f"{'='*70}")

        final_mag, final_losses = optimize_one_slice(
            'slice_0_raw',
            vocal_fps,
            mixture_mag,
            num_windows,
            CONFIG['sr'],
            current_mag=current_mag,
            iteration_offset=iteration_counter
        )
        all_losses.extend(final_losses)

        # Reconstruct audio
        print(f"\n{'='*70}")
        print("RECONSTRUCTION")
        print(f"{'='*70}")

        extracted_vocal = reconstruct_audio_from_magnitude(
            final_mag,
            mixture_phase,
            CONFIG['hop_length'],
            CONFIG['n_fft']
        )

        print("✓ Audio reconstructed and normalized")

        # Save audio
        output_audio_path = test_output_dir / "extracted_vocal.wav"
        sf.write(output_audio_path, extracted_vocal, CONFIG['sr'])
        print(f"✓ Saved: {output_audio_path}")

        # Save references (only for first test to avoid duplication)
        if num_slices == 1:
            sf.write(test_output_dir / "1_original_mixture.wav", mixture_audio, CONFIG['sr'])
            sf.write(test_output_dir / "2_target_vocal.wav", vocal_audio, CONFIG['sr'])
            print(f"✓ Saved reference files")

        # Store results
        test_duration = time.time() - test_start_time
        all_results.append({
            'num_slices': num_slices,
            'slices_used': slices_to_use + ['slice_0_raw'],
            'losses': all_losses,
            'final_loss': all_losses[-1],
            'duration': test_duration,
            'output_path': output_audio_path,
        })

        print(f"\n✓ Test {num_slices} complete in {test_duration:.1f}s")
        print(f"  Final loss: {all_losses[-1]:.6f}")
