#!/usr/bin/env python3
"""
Combine test parts into test_gain_strategies.py
"""

from pathlib import Path

# Define parts in order
parts = [
    'test_part1_header.py',
    'test_part2_helpers.py',
    'test_part3_strategies.py',
    'test_part4_main.py',
]

# Output file
output_file = Path('../test_gain_strategies.py')

print("Combining test parts into test_gain_strategies.py...")

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
            outf.write('\n\n')

print(f"\n✓ Combined file created: {output_file}")
print(f"  Total parts: {len(parts)}")

# Check file size
file_size = output_file.stat().st_size
print(f"  File size: {file_size:,} bytes")

print("\nReady to test!")
print(f"Run: python {output_file.name}")
