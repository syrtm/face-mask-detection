# Contributing to Face Mask Detection

We welcome contributions to the Face Mask Detection project! This document provides guidelines for contributing.

## How to Contribute

### Reporting Issues
- Use the GitHub issue tracker to report bugs
- Include detailed information about the issue
- Provide steps to reproduce the problem
- Include your environment details (OS, Python version, etc.)

### Submitting Changes
1. Fork the repository
2. Create a new branch for your feature (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to your branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Setup
1. Clone the repository
2. Run the setup script (`setup.bat` on Windows or `setup.sh` on Linux/macOS)
3. Make your changes
4. Test your changes thoroughly

### Code Style
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add comments for complex logic
- Keep functions small and focused

### Testing
- Test your changes before submitting
- Include tests for new features
- Ensure existing tests still pass

## Project Structure
```
face-mask-detection/
├── src/                    # Core modules
├── data/                   # Dataset (not included in repo)
├── models/                 # Trained models
├── main.py                 # Main application
├── quick_train.py          # Training script
├── quick_eval.py           # Evaluation script
├── visualize_predictions.py # Visualization
├── test_model.py           # Testing
└── requirements.txt        # Dependencies
```

## Questions?
Feel free to open an issue for any questions about contributing.
