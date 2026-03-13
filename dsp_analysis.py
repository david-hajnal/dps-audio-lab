#!/usr/bin/env python3
"""
DSP Analysis Script: DCT-II Analysis and Compression
=====================================================
This script performs:
1. Loading a mono .wav file (or generating a test signal)
2. Extracting a single window of 1024 samples
3. Applying DCT-II to generate frequency coefficients
4. Two-panel visualization (Time Domain + Frequency Domain)
5. Psychoacoustic-style compression via dB thresholding
6. Inverse DCT reconstruction with overlay on original

Author: Senior DSP Engineer
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import dct, idct
from scipy.signal import get_window
import os


def load_audio_window(filename: str, window_size: int = 1024) -> tuple:
    """
    Load a mono .wav file and extract a single window of samples.
    
    Parameters:
    -----------
    filename : str
        Path to the .wav file
    window_size : int
        Number of samples to extract (default: 1024)
    
    Returns:
    --------
    tuple : (sample_rate, x, start_index)
        - sample_rate: Sampling frequency in Hz
        - x: Array of window_size samples
        - start_index: Starting sample index in the original file
    """
    # Load the WAV file
    sample_rate, audio_data = wavfile.read(filename)
    
    # Convert stereo to mono if needed
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    
    # Normalize to [-1, 1] range if needed
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483648.0
    
    # Ensure we have enough samples
    if len(audio_data) < window_size:
        raise ValueError(f"Audio file too short: {len(audio_data)} < {window_size}")
    
    # Extract window (center it if possible)
    start_idx = (len(audio_data) - window_size) // 2
    x = audio_data[start_idx:start_idx + window_size]
    
    # Check if window is mostly silent, try to find a better window
    if np.max(np.abs(x)) < 0.01:
        print(f"  Warning: Initial window is silent, searching for better window...")
        # Search for a non-silent section
        threshold = 0.01
        for i in range(0, len(audio_data) - window_size, window_size // 2):
            segment = audio_data[i:i + window_size]
            if np.max(np.abs(segment)) > threshold:
                x = segment
                start_idx = i
                print(f"  Found better window at sample {start_idx}")
                break
    
    # Final check - if still silent, generate test signal
    if np.max(np.abs(x)) < 0.001:
        print(f"  Warning: Audio file appears to be silent. Using test signal.")
        return None, None, None
    
    return sample_rate, x, start_idx


def generate_test_signal(sample_rate: int = 44100, duration: float = 0.1, 
                         frequencies: list = None) -> tuple:
    """
    Generate a test signal with multiple sinusoidal components.
    
    Parameters:
    -----------
    sample_rate : int
        Sampling frequency in Hz
    duration : float
        Duration in seconds
    frequencies : list
        List of frequencies to generate
    
    Returns:
    --------
    tuple : (sample_rate, x)
        - sample_rate: Sampling frequency
        - x: Generated signal array (1024 samples)
    """
    if frequencies is None:
        # Complex test signal with fundamental + harmonics + noise
        frequencies = [440, 880, 1320, 2200, 3500]
    
    num_samples = 1024
    t = np.arange(num_samples) / sample_rate
    
    # Generate sinusoidal components with different amplitudes
    x = np.zeros(num_samples)
    for i, freq in enumerate(frequencies):
        # Decreasing amplitude for higher frequencies
        amplitude = 0.5 / (i + 1)
        x += amplitude * np.sin(2 * np.pi * freq * t)
    
    # Add small amount of noise
    x += 0.02 * np.random.randn(num_samples)
    
    # Apply Hann window to avoid spectral leakage
    window = get_window('hann', num_samples)
    x = x * window
    
    return sample_rate, x


def compute_dct(x: np.ndarray) -> np.ndarray:
    """
    Compute DCT-II (Discrete Cosine Transform Type II).
    
    The DCT-II is defined as:
    X[k] = sum(n=0 to N-1) x[n] * cos(pi/N * (n + 0.5) * k)
    
    Parameters:
    -----------
    x : np.ndarray
        Time-domain signal
    
    Returns:
    --------
    np.ndarray : DCT coefficients
    """
    return dct(x, type=2, norm='ortho')


def compute_idct(X: np.ndarray) -> np.ndarray:
    """
    Compute Inverse DCT-II (Discrete Cosine Transform Type II).
    
    Parameters:
    -----------
    X : np.ndarray
        DCT coefficients
    
    Returns:
    --------
    np.ndarray : Reconstructed time-domain signal
    """
    return idct(X, type=2, norm='ortho')


def compress_dct(X: np.ndarray, db_threshold: float = -40.0) -> tuple:
    """
    Compress DCT coefficients by zeroing out those below a dB threshold.
    This simulates basic psychoacoustic masking where quiet frequency 
    components are inaudible due to louder nearby components.
    
    Parameters:
    -----------
    X : np.ndarray
        DCT coefficients (magnitude)
    db_threshold : float
        Threshold in decibels (default: -40 dB)
    
    Returns:
    --------
    tuple : (X_compressed, mask)
        - X_compressed: Compressed DCT coefficients
        - mask: Boolean mask of kept coefficients
    """
    # Find the maximum magnitude
    max_magnitude = np.max(np.abs(X))
    
    if max_magnitude == 0:
        return X.copy(), np.ones_like(X, dtype=bool)
    
    # Convert to dB relative to max
    X_db = 20 * np.log10(np.abs(X) / max_magnitude + 1e-10)
    
    # Create mask: keep coefficients above threshold
    mask = X_db >= db_threshold
    
    # Apply compression
    X_compressed = X.copy()
    X_compressed[~mask] = 0
    
    # Calculate compression ratio
    kept = np.sum(mask)
    total = len(X)
    compression_ratio = (total - kept) / total * 100
    
    print(f"Compression: {kept}/{total} coefficients kept ({compression_ratio:.1f}% removed)")
    
    return X_compressed, mask


def plot_dsp_analysis(x: np.ndarray, X: np.ndarray, sample_rate: int,
                      X_compressed: np.ndarray = None, 
                      mask: np.ndarray = None,
                      db_threshold: float = -40.0):
    """
    Create two-panel visualization:
    - Top: Amplitude vs. Time (raw signal)
    - Bottom: Log-scaled Magnitude vs. Frequency (DCT coefficients)
    - Optional: Reconstructed signal overlay
    
    Parameters:
    -----------
    x : np.ndarray
        Time-domain signal (1024 samples)
    X : np.ndarray
        DCT coefficients
    sample_rate : int
        Sampling frequency in Hz
    X_compressed : np.ndarray, optional
        Compressed DCT coefficients
    mask : np.ndarray, optional
        Boolean mask of kept coefficients
    db_threshold : float
        dB threshold used for compression
    """
    n = len(x)
    freqs = np.arange(n) * sample_rate / (2 * n)  # Frequency bins for DCT
    
    # Create figure with appropriate size
    if X_compressed is not None:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    else:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # ============== TOP PANEL: Time Domain ==============
    ax1 = axes[0]
    time_axis = np.arange(n) / sample_rate * 1000  # Convert to milliseconds
    
    ax1.plot(time_axis, x, 'b-', linewidth=0.8, label='Original Signal', alpha=0.8)
    ax1.fill_between(time_axis, x, alpha=0.3)
    
    ax1.set_xlabel('Time (ms)', fontsize=11)
    ax1.set_ylabel('Amplitude', fontsize=11)
    ax1.set_title('Time Domain: Original Signal (1024 Samples)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_xlim([time_axis[0], time_axis[-1]])
    
    # Add annotation
    ax1.axhline(y=0, color='k', linewidth=0.5, linestyle='-')
    
    # ============== BOTTOM PANEL: Frequency Domain ==============
    ax2 = axes[1]
    
    # Compute magnitude spectrum
    magnitude = np.abs(X)
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    
    # Plot full spectrum
    ax2.semilogx(freqs[1:], magnitude_db[1:], 'b-', linewidth=0.8, 
                 label='DCT Magnitude', alpha=0.8)
    ax2.fill_between(freqs[1:], magnitude_db[1:], alpha=0.3)
    
    # If compression applied, show which coefficients were kept
    if mask is not None:
        # Mark kept coefficients
        ax2.scatter(freqs[1:][mask[1:]], magnitude_db[1:][mask[1:]], 
                   c='green', s=15, label=f'Kept (>{db_threshold}dB)', zorder=5)
        # Mark discarded coefficients with red x
        ax2.scatter(freqs[1:][~mask[1:]], magnitude_db[1:][~mask[1:]], 
                   c='red', s=10, marker='x', alpha=0.5, 
                   label=f'Discarded (<{db_threshold}dB)', zorder=4)
    
    ax2.set_xlabel('Frequency (Hz)', fontsize=11)
    ax2.set_ylabel('Magnitude (dB)', fontsize=11)
    ax2.set_title('Frequency Domain: DCT-II Log-Scaled Magnitude Spectrum', 
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(loc='upper right')
    ax2.set_xlim([20, sample_rate / 2])
    ax2.set_ylim([magnitude_db[1:].min() - 10, magnitude_db.max() + 5])
    
    # Add frequency labels for common audio ranges
    ax2.axvline(x=1000, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.text(1000, ax2.get_ylim()[1] - 3, '1 kHz', fontsize=8, ha='center', 
             color='gray', alpha=0.7)
    
    # ============== THIRD PANEL: Reconstruction Comparison ==============
    if X_compressed is not None:
        ax3 = axes[2]
        
        # Reconstruct signal from compressed coefficients
        x_reconstructed = compute_idct(X_compressed)
        
        # Plot original and reconstructed
        ax3.plot(time_axis, x, 'b-', linewidth=1.0, label='Original', alpha=0.8)
        ax3.plot(time_axis, x_reconstructed, 'r--', linewidth=1.0, 
                 label='Reconstructed (Compressed)', alpha=0.8)
        ax3.fill_between(time_axis, x, x_reconstructed, alpha=0.2, color='red',
                        label='Error')
        
        # Calculate SNR
        signal_power = np.mean(x ** 2)
        error = x - x_reconstructed
        noise_power = np.mean(error ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        ax3.set_xlabel('Time (ms)', fontsize=11)
        ax3.set_ylabel('Amplitude', fontsize=11)
        ax3.set_title(f'Reconstruction: Original vs Compressed (SNR = {snr:.1f} dB)', 
                      fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right')
        ax3.set_xlim([time_axis[0], time_axis[-1]])
        ax3.axhline(y=0, color='k', linewidth=0.5, linestyle='-')
    
    plt.tight_layout()
    plt.savefig('dsp_analysis_output.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved as 'dsp_analysis_output.png'")
    
    return fig


def main():
    """
    Main function to run the DSP analysis pipeline.
    """
    print("=" * 60)
    print("DCT-II Signal Analysis and Compression")
    print("=" * 60)
    
    # Configuration
    WINDOW_SIZE = 1024
    DB_THRESHOLD = -40.0  # dB threshold for psychoacoustic-like compression
    
    # Check for external wav file, otherwise generate test signal
    wav_filename = 'input.wav'
    sample_rate = None
    
    if os.path.exists(wav_filename):
        print(f"\nLoading audio from: {wav_filename}")
        sample_rate, x, start_idx = load_audio_window(wav_filename, WINDOW_SIZE)
        
        # If audio file is silent or returned None, use test signal
        if sample_rate is None:
            print(f"  Using test signal instead...")
            sample_rate, x = generate_test_signal(
                sample_rate=44100,
                duration=WINDOW_SIZE / 44100,
                frequencies=[220, 440, 880, 1320, 1760, 2200, 3300, 4400]
            )
            print(f"  Sample rate: {sample_rate} Hz")
            print(f"  Generated: {WINDOW_SIZE} samples (test signal)")
        else:
            print(f"  Sample rate: {sample_rate} Hz")
            print(f"  Window: samples {start_idx} to {start_idx + WINDOW_SIZE - 1}")
    else:
        print(f"\nNo '{wav_filename}' found. Generating test signal...")
        print(f"  Using synthetic signal with multiple frequency components")
        sample_rate, x = generate_test_signal(
            sample_rate=44100,
            duration=WINDOW_SIZE / 44100,
            frequencies=[220, 440, 880, 1320, 1760, 2200, 3300, 4400]
        )
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Generated: {WINDOW_SIZE} samples")
    
    # Signal statistics
    print(f"\n--- Signal Statistics ---")
    print(f"  Samples: {len(x)}")
    # Calculate dynamic range (handle all-zero case)
    non_zero = x[x != 0]
    if len(non_zero) > 0:
        dynamic_range = 20 * np.log10(np.max(np.abs(x)) / (np.min(np.abs(non_zero)) + 1e-10))
    else:
        dynamic_range = 0.0
    print(f"  RMS: {np.sqrt(np.mean(x**2)):.6f}")
    print(f"  Peak: {np.max(np.abs(x)):.6f}")
    print(f"  Dynamic range: {dynamic_range:.1f} dB")
    
    # Apply DCT-II
    print(f"\n--- DCT-II Transform ---")
    X = compute_dct(x)
    print(f"  DCT coefficients: {len(X)}")
    print(f"  DC component (k=0): {X[0]:.6f}")
    print(f"  Max coefficient: {np.max(np.abs(X)):.6f}")
    print(f"  Non-zero coefficients: {np.sum(np.abs(X) > 1e-10)}")
    
    # Apply compression (psychoacoustic threshold simulation)
    print(f"\n--- DCT Compression (threshold: {DB_THRESHOLD} dB) ---")
    X_compressed, mask = compress_dct(X, db_threshold=DB_THRESHOLD)
    
    # Reconstruct signal
    x_reconstructed = compute_idct(X_compressed)
    
    # Calculate metrics
    error = x - x_reconstructed
    mse = np.mean(error ** 2)
    snr = 10 * np.log10(np.mean(x**2) / (mse + 1e-10))
    
    print(f"\n--- Reconstruction Metrics ---")
    print(f"  Mean Squared Error: {mse:.8f}")
    print(f"  Signal-to-Noise Ratio: {snr:.2f} dB")
    print(f"  Max reconstruction error: {np.max(np.abs(error)):.6f}")
    
    # Create visualization
    print(f"\n--- Generating Visualization ---")
    plot_dsp_analysis(x, X, sample_rate, X_compressed, mask, DB_THRESHOLD)
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
