#!/usr/bin/env python3
"""
DCT Quantization-Based Audio Codec
===================================
A lossy audio codec using DCT with linear quantization.

Features:
1. Linear Quantization: Divide DCT coefficients by Qstep and round
2. Bitrate Simulation: Calculate sparsity ratio (% zero coefficients)
3. Quality Loop: Run with 5 different Qstep values
4. Comparative Visualization: 2x2 subplot grid
5. Audio Export: Highest and lowest quality versions
"""

import numpy as np
from scipy.fftpack import dct, idct
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os


def load_wav(filename: str) -> tuple:
    """Load a mono WAV file."""
    sample_rate, audio_data = wavfile.read(filename)
    
    # Convert stereo to mono if needed
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    
    # Normalize to [-1, 1] range
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483648.0
    
    return sample_rate, audio_data


def frame_audio(audio: np.ndarray, frame_size: int = 1024, hop_size: int = 512) -> tuple:
    """Divide audio into overlapping frames with Hann window."""
    window = np.hanning(frame_size).astype(np.float32)
    num_frames = 1 + (len(audio) - frame_size) // hop_size
    
    # Pad audio if needed
    if len(audio) < frame_size:
        audio = np.pad(audio, (0, frame_size - len(audio)))
    
    frames = np.zeros((num_frames, frame_size), dtype=np.float32)
    
    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        if end <= len(audio):
            frames[i] = audio[start:end]
        else:
            remaining = len(audio) - start
            frames[i, :remaining] = audio[start:]
    
    return frames, window


def apply_dct(frames: np.ndarray, window: np.ndarray) -> np.ndarray:
    """Apply DCT-II with analysis window."""
    num_frames = frames.shape[0]
    dct_coeffs = np.zeros_like(frames)
    
    for i in range(num_frames):
        windowed_frame = frames[i] * window
        dct_coeffs[i] = dct(windowed_frame, type=2, norm='ortho')
    
    return dct_coeffs


def linear_quantize(coeffs: np.ndarray, qstep: float) -> tuple:
    """
    Apply linear quantization to DCT coefficients.
    
    Parameters:
    -----------
    coeffs : np.ndarray
        DCT coefficients
    qstep : float
        Quantization step size
        
    Returns:
    --------
    tuple : (quantized, dequantized)
        - quantized: Integer quantized coefficients
        - dequantized: Reconstructed (dequantized) coefficients
    """
    # Quantize: divide by Qstep and round to nearest integer
    quantized = np.round(coeffs / qstep).astype(np.int32)
    
    # Dequantize: multiply back by Qstep
    dequantized = quantized.astype(np.float32) * qstep
    
    return quantized, dequantized


def calculate_sparsity(quantized: np.ndarray) -> dict:
    """
    Calculate sparsity metrics.
    
    Returns:
    --------
    dict : Sparsity statistics
    """
    total = quantized.size
    nonzero = np.sum(np.abs(quantized) > 0)
    zero = total - nonzero
    
    sparsity_ratio = zero / total * 100
    
    return {
        'total': total,
        'nonzero': nonzero,
        'zero': zero,
        'sparsity_percent': sparsity_ratio,
        'compression_ratio': nonzero / total * 100
    }


def apply_idct(coefficients: np.ndarray, window: np.ndarray) -> np.ndarray:
    """Apply inverse DCT-II with synthesis window."""
    num_frames = coefficients.shape[0]
    reconstructed = np.zeros_like(coefficients)
    
    for i in range(num_frames):
        idct_frame = idct(coefficients[i], type=2, norm='ortho')
        reconstructed[i] = idct_frame * window
    
    return reconstructed


def overlap_add(frames: np.ndarray, hop_size: int = 512, target_length: int = None) -> np.ndarray:
    """Reconstruct audio using Overlap-Add method."""
    num_frames, frame_size = frames.shape
    output_length = (num_frames - 1) * hop_size + frame_size
    
    output = np.zeros(output_length, dtype=np.float32)
    weights = np.zeros(output_length, dtype=np.float32)
    
    window = np.hanning(frame_size).astype(np.float32)
    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        output[start:end] += frames[i]
        weights[start:end] += window ** 2
    
    weights[weights == 0] = 1
    output = output / weights
    
    if target_length is not None:
        if len(output) < target_length:
            output = np.pad(output, (0, target_length - len(output)))
        else:
            output = output[:target_length]
    
    return output


def save_wav(filename: str, sample_rate: int, audio: np.ndarray):
    """Save audio as 16-bit WAV file."""
    audio_clipped = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767).astype(np.int16)
    wavfile.write(filename, sample_rate, audio_int16)


def run_quality_loop(input_file: str, qstep_values: list):
    """
    Run compression with multiple Qstep values and collect metrics.
    """
    print("=" * 70)
    print("DCT Quantization Audio Codec - Quality Analysis")
    print("=" * 70)
    
    FRAME_SIZE = 1024
    HOP_SIZE = 512
    
    # Load audio
    print(f"\n[1] Loading: {input_file}")
    sample_rate, audio = load_wav(input_file)
    original_length = len(audio)
    print(f"    Sample rate: {sample_rate} Hz")
    print(f"    Duration: {len(audio) / sample_rate:.2f} seconds")
    print(f"    Samples: {len(audio)}")
    
    # Frame audio
    print(f"\n[2] Framing: {FRAME_SIZE}-sample frames, {HOP_SIZE} hop")
    frames, window = frame_audio(audio, FRAME_SIZE, HOP_SIZE)
    num_frames = frames.shape[0]
    print(f"    Number of frames: {num_frames}")
    
    # Apply DCT
    print(f"\n[3] Transform: DCT-II with analysis window")
    dct_coeffs = apply_dct(frames, window)
    
    # Run quality loop
    print(f"\n[4] Quality Loop: Testing {len(qstep_values)} Qstep values")
    print("-" * 70)
    print(f"{'Qstep':>8} | {'Non-Zero':>10} | {'Sparsity %':>12} | {'RMSE':>12} | {'SNR (dB)':>10}")
    print("-" * 70)
    
    results = []
    
    for qstep in qstep_values:
        # Quantize
        quantized, dequantized = linear_quantize(dct_coeffs, qstep)
        
        # Calculate sparsity
        sparsity = calculate_sparsity(quantized)
        
        # Reconstruct
        reconstructed_frames = apply_idct(dequantized, window)
        reconstructed_audio = overlap_add(reconstructed_frames, HOP_SIZE, original_length)
        
        # Calculate metrics
        error = audio - reconstructed_audio
        rmse = np.sqrt(np.mean(error ** 2))
        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(error ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        result = {
            'qstep': qstep,
            'quantized': quantized,
            'dequantized': dequantized,
            'reconstructed': reconstructed_audio,
            'sparsity': sparsity,
            'rmse': rmse,
            'snr': snr
        }
        results.append(result)
        
        print(f"{qstep:>8} | {sparsity['nonzero']:>10,} | {sparsity['sparsity_percent']:>11.1f}% | {rmse:>12.6f} | {snr:>10.2f}")
    
    print("-" * 70)
    
    # Export highest and lowest quality
    print(f"\n[5] Exporting audio files...")
    
    # Lowest quality (highest Qstep = most compression)
    lowest = results[-1]
    lowest_file = f"output_qstep_{int(lowest['qstep'])}_lowest.wav"
    save_wav(lowest_file, sample_rate, lowest['reconstructed'])
    print(f"    Lowest quality: {lowest_file} (Qstep={lowest['qstep']}, SNR={lowest['snr']:.1f}dB)")
    
    # Highest quality (lowest Qstep = least compression)
    highest = results[0]
    highest_file = f"output_qstep_{int(highest['qstep'])}_highest.wav"
    save_wav(highest_file, sample_rate, highest['reconstructed'])
    print(f"    Highest quality: {highest_file} (Qstep={highest['qstep']}, SNR={highest['snr']:.1f}dB)")
    
    # Create visualization
    print(f"\n[6] Creating visualization...")
    create_visualization(audio, results, sample_rate)
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    
    return results, sample_rate


def create_visualization(original: np.ndarray, results: list, sample_rate: int):
    """
    Create 2x2 subplot visualization:
    - Plot 1: Original vs Reconstructed waveform (highest compression)
    - Plot 2: Error signal
    - Plot 3: Non-zero coefficients vs Qstep
    - Plot 4: Sparsity vs Fidelity (SNR)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('DCT Quantization Codec - Quality Analysis', fontsize=14, fontweight='bold')
    
    # Time axis for display (first 0.1 seconds)
    display_samples = min(int(0.1 * sample_rate), len(original))
    time_axis = np.arange(display_samples) / sample_rate * 1000  # ms
    
    # === Plot 1: Original vs Reconstructed (lowest quality / highest compression) ===
    ax1 = axes[0, 0]
    lowest = results[-1]
    ax1.plot(time_axis, original[:display_samples], 'b-', linewidth=0.8, 
             label='Original', alpha=0.7)
    ax1.plot(time_axis, lowest['reconstructed'][:display_samples], 'r-', 
             linewidth=0.8, label=f'Reconstructed (Q={int(lowest["qstep"])})', alpha=0.7)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'Waveform: Original vs Reconstructed (Lowest Quality)\nSNR: {lowest["snr"]:.1f} dB')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, time_axis[-1]])
    
    # === Plot 2: Error Signal ===
    ax2 = axes[0, 1]
    error = original[:display_samples] - lowest['reconstructed'][:display_samples]
    ax2.plot(time_axis, error, 'g-', linewidth=0.8, alpha=0.8)
    ax2.fill_between(time_axis, error, alpha=0.3, color='green')
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title(f'Error Signal (Original - Reconstructed)\nRMSE: {lowest["rmse"]:.6f}')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, time_axis[-1]])
    
    # === Plot 3: Non-zero coefficients vs Qstep (Bar Chart) ===
    ax3 = axes[1, 0]
    qsteps = [r['qstep'] for r in results]
    nonzero_counts = [r['sparsity']['nonzero'] for r in results]
    
    bars = ax3.bar(range(len(qsteps)), nonzero_counts, color='steelblue', alpha=0.7)
    ax3.set_xticks(range(len(qsteps)))
    ax3.set_xticklabels([str(q) for q in qsteps])
    ax3.set_xlabel('Quantization Step (Qstep)')
    ax3.set_ylabel('Non-Zero Coefficients')
    ax3.set_title('Non-Zero DCT Coefficients vs Qstep')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, nonzero_counts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000, 
                f'{val:,}', ha='center', va='bottom', fontsize=8)
    
    # === Plot 4: Sparsity vs Fidelity (Scatter) ===
    ax4 = axes[1, 1]
    sparsity = [r['sparsity']['sparsity_percent'] for r in results]
    snrs = [r['snr'] for r in results]
    rmses = [r['rmse'] for r in results]
    
    # Use RMSE for size (smaller = better)
    sizes = [100 + 500 / (r + 0.001) for r in rmses]
    
    scatter = ax4.scatter(sparsity, snrs, c=qsteps, s=sizes, cmap='RdYlGn', 
                         alpha=0.7, edgecolors='black')
    
    # Add labels for each point
    for i, (s, snr, q) in enumerate(zip(sparsity, snrs, qsteps)):
        ax4.annotate(f'Q={q}', (s, snr), textcoords="offset points", 
                    xytext=(5, 5), ha='left', fontsize=8)
    
    ax4.set_xlabel('Sparsity (% zeros)')
    ax4.set_ylabel('Signal-to-Noise Ratio (dB)')
    ax4.set_title('Sparsity vs Audio Fidelity')
    ax4.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Qstep')
    
    plt.tight_layout()
    plt.savefig('dct_quantization_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"    Visualization saved: dct_quantization_analysis.png")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DCT Quantization Audio Codec')
    parser.add_argument('input', help='Input WAV file')
    parser.add_argument('-q', '--qsteps', nargs='+', type=float, 
                       default=[1, 10, 50, 100, 500],
                       help='Qstep values to test (default: 1 10 50 100 500)')
    
    args = parser.parse_args()
    
    # Sort qstep values ascending for proper display
    qstep_values = sorted(args.qsteps)
    
    run_quality_loop(args.input, qstep_values)
