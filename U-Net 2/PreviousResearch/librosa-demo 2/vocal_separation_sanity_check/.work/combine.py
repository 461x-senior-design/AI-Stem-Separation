#!/usr/bin/env python3
"""
Combine all parts into sanity_check_progressive.py
"""

from pathlib import Path

# Define parts in order
parts = [
    'part1_header_cli.py',
    'part2_helpers.py',
    'part3_fingerprints.py',
    'part4_optimization.py',
    'part5_main_loop.py',
    'part6_visualization.py',
]

# Output file
output_file = Path('../sanity_check_progressive.py')

print("Combining parts into sanity_check_progressive.py...")

with open(output_file, 'w') as outf:
    for part_name in parts:
        part_path = Path(part_name)

        if not part_path.exists():
            print(f"  ERROR: {part_name} not found!")
            exit(1)

        print(f"  Adding {part_name}...")

        with open(part_path, 'r') as inf:
            content = inf.read()
            outf.write(content)
            outf.write('\n\n')  # Add spacing between parts

print(f"\n✓ Combined file created: {output_file}")
print(f"  Total parts: {len(parts)}")

# Check file size
file_size = output_file.stat().st_size
print(f"  File size: {file_size:,} bytes")

print("\nReady to use!")
print(f"Run: python {output_file.name} --slices 18")
