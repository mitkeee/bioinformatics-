#!/bin/bash

# Setup git configuration
git config user.name "mitkeee"
git config user.email "user@example.com"

# Add all files
git add .

# Commit changes
git commit -m "Final protein burial analysis system with DSSP/STRIDE integration and confusion matrix metrics" || echo "Already committed"

# Ensure main branch
git branch -M main

# Push to GitHub
git push -u origin main

echo "✅ Code pushed to GitHub successfully!"
echo "Repository: https://github.com/mitkeee/bioinformatics-"

