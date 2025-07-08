#!/bin/bash

# Git Repository Setup Script
echo "Setting up Git repository for Face Mask Detection..."

# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Face mask detection project

- Added core detection scripts
- Added training and evaluation modules
- Added visualization tools
- Added comprehensive documentation
- Added setup scripts for easy installation"

echo "Git repository initialized!"
echo ""
echo "Next steps:"
echo "1. Create a GitHub repository"
echo "2. Add remote origin: git remote add origin https://github.com/yourusername/face-mask-detection.git"
echo "3. Push to GitHub: git push -u origin main"
echo ""
echo "Optional: Create branches for development"
echo "git checkout -b development"
