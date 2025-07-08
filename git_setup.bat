@echo off
echo Setting up Git repository for Face Mask Detection...

rem Initialize git repository
git init

rem Add all files
git add .

rem Create initial commit
git commit -m "Initial commit: Face mask detection project" -m "- Added core detection scripts" -m "- Added training and evaluation modules" -m "- Added visualization tools" -m "- Added comprehensive documentation" -m "- Added setup scripts for easy installation"

echo.
echo Git repository initialized!
echo.
echo Next steps:
echo 1. Create a GitHub repository
echo 2. Add remote origin: git remote add origin https://github.com/yourusername/face-mask-detection.git
echo 3. Push to GitHub: git push -u origin main
echo.
echo Optional: Create branches for development
echo git checkout -b development

pause
