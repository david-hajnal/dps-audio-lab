#!/bin/bash
#
# Setup script for DSP Analysis Environment
# Creates a Python virtual environment and installs dependencies
#

set -e

echo "============================================"
echo "DSP Analysis Environment Setup"
echo "============================================"

# Create virtual environment
echo "Creating virtual environment 'dsp-env'..."
python3 -m venv dsp-env

# Activate virtual environment
echo "Activating virtual environment..."
source dsp-env/bin/activate

# Install requirements
echo "Installing packages..."
pip install -r requirements.txt

echo ""
echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "To run the DSP analysis script:"
echo "  source dsp-env/bin/activate"
echo "  python dsp_analysis.py"
echo ""
