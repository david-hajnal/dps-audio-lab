#!/usr/bin/env python3
"""
DCT-Based Audio Codec
=====================
A basic lossy audio codec using DCT-II compression.

Operations:
1. Load mono WAV file
2. Frame audio into overlapping blocks (1024 samples, 50% overlap, Hann window)
3. Apply DCT-II to each block
4. Compress using threshold-based coefficient elimination
5. Apply IDCT to each block
6. Reconstruct using Overlap-Add method
7. Export as new WAV file with compression statistics
"""

import numpy as np
from scipy.fftpack import dct, idct
from scipy.io import wavfile
import os


def load_wav(filename: str) -> tuple:
    """
    Load a mono WAV file.
    
    Parameters:
    -----------
    filename : str
        Path to the WAV file
        
    Returns:
    --------
    tuple : (sample_rate, audio_data)
        - sample_rate: Sampling frequency in Hz
        - audio_data: Normalized float32 audio samples
    """
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
    """
    Divide audio into overlapping frames.
    
    Parameters:
    -----------
    audio : np.ndarray
        Input audio samples
    frame_size : int
        Number of samples per frame (default: 1024)
    hop_size : int
        Number of samples between consecutive frames (default: 512 = 50% overlap)
        
    Returns:
    --------
    tuple : (frames, window)
        - frames: 2D array of shape (num_frames, frame_size)
        - window: The window function applied to each frame
    """
    # Create Hanning window
    window = np.hanning(frame_size).astype(np.float32)
    
    # Calculate number of frames needed
    num_frames = 1 + (len(audio) - frame_size) // hop_size
    
    # Pad audio if needed
    if len(audio) < frame_size:
        audio = np.pad(audio, (0, frame_size - len(audio)))
    
    # Extract frames
    frames = np.zeros((num_frames, frame_size), dtype=np.float32)
    
    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        if end <= len(audio):
            frames[i] = audio[start:end]
        else:
            # Handle last frame with padding
            remaining = len(audio) - start
            frames[i, :remaining] = audio[start:]
            frames[i, remaining:] = 0  # Zero padding
    
    return frames, window


def apply_dct(frames: np.ndarray, window: np.ndarray) -> np.ndarray:
    """
    Apply DCT-II to each frame (with analysis window applied).
    
    Parameters:
    -----------
    frames : np.ndarray
        2D array of audio frames
    window : np.ndarray
        Window function (applied as analysis window)
        
    Returns:
    --------
    np.ndarray : DCT coefficients for each frame
    """
    num_frames = frames.shape[0]
    dct_coeffs = np.zeros_like(frames)
    
    for i in range(num_frames):
        # Apply analysis window before DCT
        windowed_frame = frames[i] * window
        dct_coeffs[i] = dct(windowed_frame, type=2, norm='ortho')
    
    return dct_coeffs


def threshold_compression(coeffs: np.ndarray, db_limit: float) -> tuple:
    """
    Compress DCT coefficients by zeroing out those below a dB threshold.
    
    This simulates basic psychoacoustic compression where quiet frequency
    components are removed relative to the peak coefficient.
    
    Parameters:
    -----------
    coeffs : np.ndarray
        2D array of DCT coefficients
    db_limit : float
        Threshold in decibels (relative to peak). All coefficients below
        this threshold will be zeroed out.
        
    Returns:
    --------
    tuple : (compressed_coeffs, stats)
        - compressed_coeffs: DCT coefficients after compression
        - stats: Dictionary with compression statistics
    """
    compressed = coeffs.copy()
    original_nonzero = np.sum(np.abs(compressed) > 1e-10)
    
    # Find peak magnitude across all frames
    peak_magnitude = np.max(np.abs(compressed))
    
    if peak_magnitude == 0:
        return compressed, {'kept': 0, 'total': compressed.size, 'ratio': 0}
    
    # Convert to dB relative to peak
    # Threshold amplitude = peak * 10^(db_limit/20)
    threshold = peak_magnitude * (10 ** (db_limit / 20))
    
    # Create mask for coefficients above threshold
    mask = np.abs(compressed) >= threshold
    
    # Zero out coefficients below threshold
    compressed[~mask] = 0
    
    # Calculate statistics
    kept_nonzero = np.sum(np.abs(compressed) > 1e-10)
    total_coeffs = compressed.size
    compression_ratio = (total_coeffs - kept_nonzero) / total_coeffs * 100
    
    stats = {
        'kept': kept_nonzero,
        'total': total_coeffs,
        'ratio': compression_ratio,
        'threshold': threshold,
        'peak': peak_magnitude
    }
    
    return compressed, stats


def apply_idct(coefficients: np.ndarray, window: np.ndarray) -> np.ndarray:
    """
    Apply inverse DCT-II to each frame (with synthesis window applied).
    
    Parameters:
    -----------
    coefficients : np.ndarray
        2D array of DCT coefficients
    window : np.ndarray
        Window function (applied as synthesis window)
        
    Returns:
    --------
    np.ndarray : Reconstructed time-domain frames
    """
    num_frames = coefficients.shape[0]
    reconstructed = np.zeros_like(coefficients)
    
    for i in range(num_frames):
        # Apply IDCT then synthesis window
        idct_frame = idct(coefficients[i], type=2, norm='ortho')
        reconstructed[i] = idct_frame * window
    
    return reconstructed


def overlap_add(frames: np.ndarray, hop_size: int = 512, 
                   target_length: int = None) -> np.ndarray:
    """
    Reconstruct audio using Overlap-Add method.
    
    Note: Windowing is already applied in the IDCT step.
    
    Parameters:
    -----------
    frames : np.ndarray
        2D array of time-domain frames (window already applied)
    hop_size : int
        Number of samples between consecutive frames
    target_length : int, optional
        Target output length (for trimming/padding)
        
    Returns:
    --------
    np.ndarray : Reconstructed audio signal
    """
    num_frames, frame_size = frames.shape
    
    # Calculate output length
    output_length = (num_frames - 1) * hop_size + frame_size
    
    # Initialize output and accumulator
    output = np.zeros(output_length, dtype=np.float32)
    weights = np.zeros(output_length, dtype=np.float32)
    
    # Apply window and overlap-add
    window = np.hanning(frame_size).astype(np.float32)
    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        
        # Add to output with overlap (window already applied to frames)
        output[start:end] += frames[i]
        weights[start:end] += window ** 2  # Accumulate window energy
    
    # Normalize by accumulated weights (handle windowing effects)
    # Avoid division by zero
    weights[weights == 0] = 1
    output = output / weights
    
    # Trim or pad to target length
    if target_length is not None:
        if len(output) < target_length:
            output = np.pad(output, (0, target_length - len(output)))
        else:
            output = output[:target_length]
    
    return output


def save_wav(filename: str, sample_rate: int, audio: np.ndarray):
    """
    Save audio as 16-bit WAV file.
    
    Parameters:
    -----------
    filename : str
        Output filename
    sample_rate : int
        Sampling frequency in Hz
    audio : np.ndarray
        Audio samples (normalized to [-1, 1])
    """
    # Clip to valid range and convert to int16
    audio_clipped = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767).astype(np.int16)
    
    wavfile.write(filename, sample_rate, audio_int16)
    print(f"Saved: {filename}")


def dct_codec(input_file: str, output_file: str, db_limit: float = -40.0):
    """
    Main DCT codec function.
    
    Parameters:
    -----------
    input_file : str
        Input WAV filename
    output_file : str
        Output WAV filename
    db_limit : float
        Compression threshold in dB (default: -40)
    """
    print("=" * 60)
    print("DCT Audio Codec")
    print("=" * 60)
    
    # Configuration
    FRAME_SIZE = 1024
    HOP_SIZE = 512  # 50% overlap
    
    # Load audio
    print(f"\n[1] Loading: {input_file}")
    sample_rate, audio = load_wav(input_file)
    print(f"    Sample rate: {sample_rate} Hz")
    print(f"    Duration: {len(audio) / sample_rate:.2f} seconds")
    print(f"    Samples: {len(audio)}")
    
    # Frame audio
    print(f"\n[2] Framing: {FRAME_SIZE}-sample frames, {HOP_SIZE} hop ({50}% overlap)")
    frames, window = frame_audio(audio, FRAME_SIZE, HOP_SIZE)
    num_frames = frames.shape[0]
    print(f"    Number of frames: {num_frames}")
    
    # Apply DCT
    print(f"\n[3] Transform: Applying DCT-II to each frame")
    dct_coeffs = apply_dct(frames, window)
    print(f"    DCT coefficients per frame: {dct_coeffs.shape[1]}")
    
    # Compress
    print(f"\n[4] Compress: Threshold = {db_limit} dB")
    compressed_coeffs, stats = threshold_compression(dct_coeffs, db_limit)
    print(f"    Coefficients kept: {stats['kept']:,} / {stats['total']:,}")
    print(f"    Compression ratio: {stats['ratio']:.1f}%")
    print(f"    Peak magnitude: {stats['peak']:.6f}")
    print(f"    Threshold value: {stats['threshold']:.6f}")
    
    # Apply IDCT
    print(f"\n[5] Inverse Transform: Applying IDCT-II to each frame")
    reconstructed_frames = apply_idct(compressed_coeffs, window)
    
    # Overlap-Add
    print(f"\n[6] Reconstruct: Overlap-Add method")
    reconstructed_audio = overlap_add(reconstructed_frames, HOP_SIZE, len(audio))
    
    # Calculate compression ratio (non-zero coefficients vs original samples)
    original_sample_count = len(audio)
    compression_ratio_final = stats['kept'] / original_sample_count
    
    # Save output
    print(f"\n[7] Export: Saving to {output_file}")
    save_wav(output_file, sample_rate, reconstructed_audio)
    
    # Calculate and display final statistics
    print(f"\n--- Compression Statistics ---")
    print(f"  Original samples: {original_sample_count:,}")
    print(f"  Non-zero DCT coefficients: {stats['kept']:,}")
    print(f"  Compression ratio: {compression_ratio_final:.2%}")
    print(f"  Compression: {100 - compression_ratio_final * 100:.1f}% reduction")
    
    # Calculate SNR
    error = audio - reconstructed_audio
    signal_power = np.mean(audio ** 2)
    noise_power = np.mean(error ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    print(f"  Signal-to-Noise Ratio: {snr:.2f} dB")
    
    print("\n" + "=" * 60)
    print("Encoding Complete!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DCT-based Audio Codec')
    parser.add_argument('input', help='Input WAV file')
    parser.add_argument('-o', '--output', help='Output WAV file (default: input_compressed.wav)')
    parser.add_argument('-d', '--db', type=float, default=-40.0, 
                       help='Compression threshold in dB (default: -40)')
    
    args = parser.parse_args()
    
    # Set default output filename
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_compressed.wav"
    
    # Run codec
    dct_codec(args.input, args.output, args.db)
