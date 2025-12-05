#!/usr/bin/env python3
import subprocess
import sys

pdb_path = "/Users/famnit/Desktop/pythonProject/dude_extracted/dude_1_2/igf1r/receptor.pdb"

try:
    result = subprocess.run(['stride', pdb_path], capture_output=True, text=True, timeout=30)

    with open('/tmp/stride_test_output.txt', 'w') as f:
        f.write("=== STRIDE STDOUT ===\n")
        f.write(f"Return code: {result.returncode}\n\n")
        f.write(result.stdout[:5000])  # First 5000 chars
        f.write("\n\n=== STRIDE STDERR ===\n")
        f.write(result.stderr[:1000])

    print("Output written to /tmp/stride_test_output.txt")

    # Also count ASG lines
    asg_count = sum(1 for line in result.stdout.split('\n') if line.startswith('ASG'))
    print(f"ASG lines found: {asg_count}")

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

