#!/usr/bin/env python3
"""Test STRIDE parsing to debug the issue."""

from pathlib import Path

def parse_stride_file_test(stride_path: Path):
    """Test parse function with debugging output."""
    stride_data = {}
    seq_position = 0
    asg_count = 0

    if not stride_path.exists():
        print(f"File not found: {stride_path}")
        return stride_data

    with open(stride_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.startswith('ASG'):
                asg_count += 1
                parts = line.split()

                if asg_count <= 3:
                    print(f"ASG record {asg_count}: {len(parts)} parts")
                    print(f"  Full line: {line[:100]}")
                    print(f"  Parts: {parts}")

                if len(parts) < 10:
                    print(f"  Skipped (< 10 parts)")
                    continue

                try:
                    resname = parts[1]
                    ss = parts[5]
                    asa_value = parts[-1]
                    asa = float(asa_value)

                    seq_position += 1
                    stride_data[seq_position] = {
                        'resname': resname,
                        'ss': ss,
                        'asa': asa
                    }

                    if seq_position <= 3:
                        print(f"  Parsed OK: seq_pos={seq_position}, resname={resname}, ss={ss}, asa={asa}")

                except Exception as e:
                    print(f"  Parse error: {e}")

    print(f"\nTotal ASG records found: {asg_count}")
    print(f"Total records parsed: {len(stride_data)}")
    print(f"\nFirst 5 parsed records:")
    for i in range(1, 6):
        if i in stride_data:
            data = stride_data[i]
            print(f"  {i}: {data['resname']} SS={data['ss']} ASA={data['asa']}")

    return stride_data

# Test with abl1
stride_path = Path("holder/dude_1_2/abl1/receptor.stride")
print(f"Testing {stride_path}\n")
data = parse_stride_file_test(stride_path)

