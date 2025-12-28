#!/usr/bin/env python3
"""
Convert multi-line training data format to single-line format.

Old format:
### Input:
Add node cache

### Output:
ADD_NODE cache

New format:
### Input: Add node cache ### Output: ADD_NODE cache
"""

import sys
from pathlib import Path


def convert_file(input_path):
    """Convert a multi-line format file to single-line format."""
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by "### Input:" to get individual examples
    examples = content.split('### Input:')[1:]  # Skip empty first element

    single_line_examples = []

    for example in examples:
        # Split by "### Output:" to separate input and output
        parts = example.split('### Output:')
        if len(parts) != 2:
            continue

        input_text = parts[0].strip()
        output_text = parts[1].strip()

        if not input_text or not output_text:
            continue

        # Create single-line format
        # Convert newlines in output to " | " delimiter
        output_single_line = output_text.replace('\n', ' | ')
        single_line = f"### Input: {input_text} ### Output: {output_single_line}"
        single_line_examples.append(single_line)

    # Write back to file
    output_content = '\n'.join(single_line_examples)
    with open(input_path, 'w', encoding='utf-8') as f:
        f.write(output_content)

    print(f"✅ Converted {len(single_line_examples)} examples in {input_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_to_single_line.py <file1> <file2> ...")
        sys.exit(1)

    for file_path in sys.argv[1:]:
        convert_file(Path(file_path))
