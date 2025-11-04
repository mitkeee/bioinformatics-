# Installing STRIDE for macOS

## Quick Install Guide

STRIDE is a program for protein secondary structure assignment from atomic coordinates.

### Method 1: Download Pre-compiled Binary (Easiest)

1. Visit: http://webclu.bio.wzw.tum.de/stride/
2. Download the macOS binary
3. Make it executable and move to your PATH:
   ```bash
   chmod +x stride
   sudo mv stride /usr/local/bin/
   ```

### Method 2: Compile from Source

```bash
# Download source code
wget http://webclu.bio.wzw.tum.de/stride/stride.tar.gz
tar -xzf stride.tar.gz
cd stride

# Compile
make

# Install
sudo cp stride /usr/local/bin/
```

### Method 3: Use Homebrew (if available)

```bash
# If you have a custom tap or formula
brew install stride
```

### Verify Installation

```bash
which stride
stride -h
```

## After Installing STRIDE

Run the generation script:
```bash
cd /Users/famnit/Desktop/pythonProject
python generate_stride_files.py
```

This will create `.stride` files with ASG records containing:
- Accessible Surface Area (ASA) in Ų
- Secondary structure assignment (H, E, C, etc.)
- Phi/Psi angles

Then re-run the analysis:
```bash
python protein_burial_analysis.py
python generate_detailed_reports.py
```

