# DSP Audio Lab

A collection of Python DSP tools for audio processing, analysis, and creative sound design.

## Overview

This repository contains various audio processing scripts using DCT (Discrete Cosine Transform) techniques for both analytical and creative applications.

## Prerequisites

- Python 3.x
- macOS (or compatible Unix-like system)

## Repository Structure

```
dsp-audio-lab/
├── README.md
├── setup_env.sh
├── requirements.txt
├── .gitignore
├── audio_samples/       # Input WAV files
│   └── input.wav
├── dsp_analysis.py
├── dct_codec.py
├── dct_quantization_codec.py
└── spectral_artifact_generator.py
```

## Quick Start

```bash
# Setup virtual environment
./setup_env.sh

# Activate environment
source dsp-env/bin/activate
```

## Scripts

### 1. dsp_analysis.py
Basic DSP analysis with DCT visualization.

**Features:**
- Load WAV files and extract 1024-sample windows
- Apply DCT-II transform
- Two-panel visualization (Time Domain + Frequency Domain)
- Compression via dB threshold
- Reconstruction with SNR metrics

**Usage:**
```bash
python dsp_analysis.py
```

### 2. dct_codec.py
DCT-based lossy audio codec.

**Features:**
- Overlapping frame analysis (1024 samples, 50% hop)
- Hann windowing
- Threshold-based coefficient compression
- Overlap-Add reconstruction

**Usage:**
```bash
python dct_codec.py input.wav -d -80
```

### 3. dct_quantization_codec.py
Quantization-based compression with quality analysis.

**Features:**
- Linear quantization with configurable Qstep
- Quality loop with multiple Qstep values
- 2x2 visualization grid (waveforms, error, coefficients, SNR)
- Exports highest/lowest quality WAV files

**Usage:**
```bash
python dct_quantization_codec.py input.wav
```

### 4. spectral_artifact_generator.py
Creative "illegal" DSP effects for sound design.

**Features:**
- Non-linear quantization (sine/tan harmonic folding)
- Spectral gating & freezing (shimmering artifacts)
- Spectral smear (bin shifting for alien sounds)
- Block-size modulation (rhythmic jitter)
- 10-second morph effect

**Usage:**
```bash
# Create 10-second morph effect
python spectral_artifact_generator.py input.wav -d 10

# Demo all individual effects
python spectral_artifact_generator.py input.wav --demo
```

## Output Files

Generated files are automatically created in the project root. All WAV files are ignored by git (see `.gitignore`).

| File | Description |
|------|-------------|
| `audio_samples/input.wav` | Source audio (place your file here) |
| `dsp_analysis_output.png` | DCT analysis visualization |
| `input_compressed.wav` | Threshold-compressed audio |
| `output_qstep_1_highest.wav` | Best quality quantization |
| `output_qstep_500_lowest.wav` | Lowest quality quantization |
| `dct_quantization_analysis.png` | Quality analysis plots |
| `spectral_morph.wav` | 10-sec artifact morph |
| `effect_*.wav` | Individual effect demos |

## Requirements

```
numpy
scipy
matplotlib
```

## License

MIT
