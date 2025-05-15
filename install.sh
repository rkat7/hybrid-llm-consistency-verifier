#!/bin/bash
# Installation script for NLP-ASP system

echo "===================================="
echo "Installing NLP-ASP System"
echo "===================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not found. Please install Python 3 and try again."
    exit 1
fi

# Create virtual environment (if not exists)
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Download SpaCy model
echo "Downloading SpaCy model..."
python -m spacy download en_core_web_sm

# Create necessary directories if they don't exist
echo "Setting up directories..."
mkdir -p results/test_outputs results/advanced

echo "===================================="
echo "Installation complete!"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run a simple example:"
echo "  python src/simple_example.py data/example_inputs/input.txt"
echo "====================================" 