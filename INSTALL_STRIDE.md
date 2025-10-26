# Installing STRIDE on macOS

STRIDE is a program for protein secondary structure assignment and solvent accessibility calculation.

## Method 1: Download Pre-compiled Binary (Easiest)

1. Download STRIDE from: http://webclu.bio.wzw.tum.de/stride/
2. Or use this direct link for the source: ftp://ftp.ebi.ac.uk/pub/software/unix/stride/

## Method 2: Compile from Source

### Step 1: Download STRIDE source code

```bash
cd ~/Downloads
curl -O ftp://ftp.ebi.ac.uk/pub/software/unix/stride/src/stride.tar.gz
tar -xzf stride.tar.gz
cd stride
```

### Step 2: Compile

```bash
make
```

### Step 3: Install to system path

```bash
sudo cp stride /usr/local/bin/
sudo chmod +x /usr/local/bin/stride
```

### Step 4: Verify installation

```bash
stride -h
```

You should see STRIDE help information.

## Method 3: Using Homebrew (if available)

Some users have created homebrew formulas, but STRIDE is not in the official homebrew repository.

## After Installation

Once STRIDE is installed, your Python code will automatically detect it and run STRIDE validation when you execute:

```bash
python extract_ca.py
```

The output will include:
- DSSP agreement (already working)
- STRIDE agreement (will work after installation)
- DSSP vs STRIDE comparison (shows how much these two methods agree)

## Troubleshooting

If `stride -h` doesn't work after installation:
1. Make sure `/usr/local/bin` is in your PATH
2. Check with: `echo $PATH`
3. Add to PATH if needed: `export PATH="/usr/local/bin:$PATH"`

## Alternative: Test Without Installing

You can disable STRIDE validation by setting:

```python
DO_STRIDE = False  # in extract_ca.py at the bottom
```

The code will still run with just DSSP validation.

