#!/usr/bin/env python3
"""
Spectral Artifact Generator
===========================
A creative DSP tool that creates "illegal" audio artifacts through
non-standard spectral processing techniques.

WARNING: These are intentionally "broken" DSP operations designed for
sound design and creative audio effects - NOT for audio compression!

Effects:
1. Non-Linear Quantization: Sine/tan harmonic folding
2. Spectral Gating & Freezing: Random coefficient manipulation
3. Smear Effect: Bin shifting for alien metallic sounds
4. Block-Size Modulation: Random frame sizes for jitter
5. 10-second morph from clean to deconstructed
"""

import numpy as np
from scipy.fftpack import dct, idct
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os


def load_wav(filename: str) -> tuple:
    """Load a mono WAV file."""
    sample_rate, audio_data = wavfile.read(filename)
    
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483648.0
    
    return sample_rate, audio_data


def nonlinear_quantize(coeffs: np.ndarray, qstep: float, mode: str = 'sine') -> np.ndarray:
    """
    Non-linear quantization using sine or tan functions.
    Creates harmonic folding artifacts.
    """
    if mode == 'sine':
        # Apply sine compression - creates harmonic distortion
        # Large coefficients get folded back into lower ranges
        compressed = np.sin(coeffs / qstep * np.pi / 2)
        quantized = np.round(compressed).astype(np.int32)
        dequantized = np.sin(quantized / np.pi * 2) * (qstep / (np.pi / 2))
    elif mode == 'tan':
        # Tangent creates extreme folding at boundaries
        compressed = np.tanh(coeffs / qstep * 3)
        quantized = np.round(compressed * 1000).astype(np.int32)
        dequantized = np.arctan(quantized / 1000) * (qstep / 3)
    else:
        # Fallback to linear
        quantized = np.round(coeffs / qstep).astype(np.int32)
        dequantized = quantized.astype(np.float32) * qstep
    
    return dequantized


def spectral_gate_freeze(coeffs: np.ndarray, freeze_prob: float = 0.1, 
                        gain_range: tuple = (0, 10)) -> np.ndarray:
    """
    Randomly zero or amplify coefficients to create shimmering/gurgling.
    
    Parameters:
    -----------
    freeze_prob : float
        Probability of setting coefficient to zero (gating)
    gain_range : tuple
        Range for random gain multiplier (shimmering)
    """
    result = coeffs.copy()
    
    # Random gating (zeroing)
    gate_mask = np.random.random(coeffs.shape) < freeze_prob
    result[gate_mask] = 0
    
    # Random gain (freezing/shimmering)
    freeze_mask = np.random.random(coeffs.shape) < freeze_prob
    random_gains = np.random.uniform(gain_range[0], gain_range[1], coeffs.shape)
    result[freeze_mask] *= random_gains[freeze_mask]
    
    return result


def spectral_smear(coeffs: np.ndarray, shift: int = 1) -> np.ndarray:
    """
    Shift DCT coefficients by N bins before IDCT.
    Creates alien, metallic, pitch-shifted artifacts.
    
    Parameters:
    -----------
    shift : int
        Number of bins to shift (positive = up, negative = down)
    """
    result = np.zeros_like(coeffs)
    
    # Handle both 1D and 2D arrays
    if coeffs.ndim == 1:
        if shift > 0:
            result[shift:] = coeffs[:-shift]
            result[:shift] = coeffs[0]
        else:
            shift = abs(shift)
            result[:-shift] = coeffs[shift:]
            result[-shift:] = coeffs[-1]
    else:
        # 2D array (multiple frames)
        if shift > 0:
            result[:, shift:] = coeffs[:, :-shift]
            result[:, :shift] = coeffs[:, 0:1]
        else:
            shift = abs(shift)
            result[:, :-shift] = coeffs[:, shift:]
            result[:, -shift:] = coeffs[:, -1:]
    
    return result


def apply_dct_variable(frames: np.ndarray, window: np.ndarray) -> list:
    """Apply DCT to frames - returns list to handle variable sizes."""
    dct_coeffs = []
    for frame in frames:
        coeff = dct(frame * window, type=2, norm='ortho')
        dct_coeffs.append(coeff)
    return dct_coeffs


def apply_idct_variable(dct_coeffs: list, window: np.ndarray) -> list:
    """Apply IDCT to coefficient list."""
    reconstructed = []
    for coeff in dct_coeffs:
        frame = idct(coeff, type=2, norm='ortho')
        reconstructed.append(frame * window)
    return reconstructed


def block_size_modulation(audio: np.ndarray, sample_rate: int, 
                         min_size: int = 64, max_size: int = 4096,
                         hop_ratio: float = 0.5) -> tuple:
    """
    Split audio into blocks of randomized sizes.
    Creates rhythmic jitter and variable resolution.
    """
    window = np.hanning(max_size).astype(np.float32)
    
    blocks = []
    positions = []
    pos = 0
    
    while pos < len(audio):
        # Random block size between min and max
        block_size = np.random.randint(min_size, max_size + 1)
        
        # Adjust hop size (50% of block size by default)
        hop_size = int(block_size * hop_ratio)
        
        # Ensure we don't go past the end
        end = min(pos + block_size, len(audio))
        block = audio[pos:end]
        
        # Pad if necessary
        if len(block) < max_size:
            block = np.pad(block, (0, max_size - len(block)))
        
        # Apply window
        windowed = block * window
        
        blocks.append((windowed, block_size))
        positions.append((pos, end))
        
        pos += hop_size
    
    return blocks, positions, window


def process_segment(frames, window, effect_params, segment_num, total_segments):
    """Process a segment with a specific effect."""
    dct_coeffs = apply_dct_variable(frames, window)
    
    processed = []
    for i, coeff in enumerate(dct_coeffs):
        # Non-linear quantization (always applied)
        if effect_params.get('nonlinear', False):
            mode = effect_params.get('mode', 'sine')
            qstep = effect_params.get('qstep', 10)
            coeff = nonlinear_quantize(coeff, qstep, mode)
        
        # Spectral gating/freezing
        if effect_params.get('gating', False):
            freeze_prob = effect_params.get('freeze_prob', 0.1)
            gain_range = effect_params.get('gain_range', (0, 10))
            coeff = spectral_gate_freeze(coeff, freeze_prob, gain_range)
        
        # Spectral smear
        if effect_params.get('smear', False):
            shift = effect_params.get('smear_shift', 1)
            coeff = spectral_smear(coeff, shift)
        
        processed.append(coeff)
    
    # Reconstruct
    reconstructed_frames = apply_idct_variable(processed, window)
    
    return reconstructed_frames


def create_morph_effect(input_file: str, output_file: str, duration: float = 10.0):
    """
    Create a 10-second morph from clean audio to deconstructed artifacts.
    """
    print("=" * 70)
    print("Spectral Artifact Generator - Morph Effect")
    print("=" * 70)
    
    FRAME_SIZE = 1024
    HOP_SIZE = 512
    
    # Load audio
    print(f"\n[1] Loading: {input_file}")
    sample_rate, audio = load_wav(input_file)
    print(f"    Sample rate: {sample_rate} Hz")
    
    # Target duration (10 seconds or shorter if input is shorter)
    target_samples = min(int(duration * sample_rate), len(audio))
    audio = audio[:target_samples]
    print(f"    Target duration: {target_samples / sample_rate:.2f} seconds")
    
    # Frame audio
    print(f"\n[2] Processing: Creating morph effect over {duration} seconds")
    print(f"    Frame size: {FRAME_SIZE}, Hop: {HOP_SIZE}")
    
    # Calculate number of frames
    num_frames = 1 + (len(audio) - FRAME_SIZE) // HOP_SIZE
    
    # Create Hann window
    window = np.hanning(FRAME_SIZE).astype(np.float32)
    
    # Extract frames
    frames = []
    for i in range(num_frames):
        start = i * HOP_SIZE
        end = start + FRAME_SIZE
        if end <= len(audio):
            frames.append(audio[start:end])
        else:
            frame = np.zeros(FRAME_SIZE)
            frame[:len(audio) - start] = audio[start:]
            frames.append(frame)
    
    frames = np.array(frames)
    
    # Process each frame with increasing effect intensity
    print("\n[3] Applying spectral effects...")
    
    # Process all frames
    dct_coeffs = apply_dct_variable(frames, window)
    
    processed_frames = []
    
    for i, coeff in enumerate(dct_coeffs):
        # Calculate progress (0 to 1)
        progress = i / max(num_frames - 1, 1)
        
        # Interpolate effect parameters based on progress
        # 0% = clean, 100% = fully deconstructed
        
        # Non-linear quantization (always increases)
        qstep = 1 + progress * 99  # 1 to 100
        if progress > 0.1:
            coeff = nonlinear_quantize(coeff, qstep, 'sine')
        
        # Spectral gating (starts at 30%)
        if progress > 0.3:
            freeze_prob = (progress - 0.3) * 0.5  # 0 to 0.35
            gain_range = (0, 1 + progress * 20)
            coeff = spectral_gate_freeze(coeff, freeze_prob, gain_range)
        
        # Spectral smear (starts at 50%)
        if progress > 0.5:
            shift = int((progress - 0.5) * 20)  # 0 to 10
            if shift > 0:
                coeff = spectral_smear(coeff, shift)
        
        # Reconstruct
        reconstructed = idct(coeff, type=2, norm='ortho')
        processed_frames.append(reconstructed * window)
    
    processed_frames = np.array(processed_frames)
    
    # Overlap-Add reconstruction
    print("[4] Reconstructing audio...")
    output_length = (num_frames - 1) * HOP_SIZE + FRAME_SIZE
    output = np.zeros(output_length, dtype=np.float32)
    weights = np.zeros(output_length, dtype=np.float32)
    
    for i in range(num_frames):
        start = i * HOP_SIZE
        end = start + FRAME_SIZE
        output[start:end] += processed_frames[i]
        weights[start:end] += window ** 2
    
    weights[weights == 0] = 1
    output = output / weights
    
    # Trim to target
    output = output[:target_samples]
    
    # Save output
    print(f"\n[5] Saving: {output_file}")
    audio_clipped = np.clip(output, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767).astype(np.int16)
    wavfile.write(output_file, sample_rate, audio_int16)
    
    # Create visualization
    print("[6] Creating visualization...")
    create_visualization(audio, output, sample_rate)
    
    print("\n" + "=" * 70)
    print("Spectral Artifact Generation Complete!")
    print("=" * 70)


def create_visualization(original: np.ndarray, processed: np.ndarray, sample_rate: int):
    """Create before/after visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Spectral Artifact Generator - Morph Effect', fontsize=14, fontweight='bold')
    
    # Display first 0.05 seconds
    display = int(0.05 * sample_rate)
    time_axis = np.arange(display) / sample_rate * 1000
    
    # Plot 1: Original waveform
    ax1 = axes[0, 0]
    ax1.plot(time_axis, original[:display], 'b-', linewidth=0.8)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Original Audio (Clean)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, time_axis[-1]])
    
    # Plot 2: Processed waveform
    ax2 = axes[0, 1]
    ax2.plot(time_axis, processed[:display], 'r-', linewidth=0.8)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Processed Audio (Deconstructed)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, time_axis[-1]])
    
    # Plot 3: Spectrogram of original
    ax3 = axes[1, 0]
    # Simple magnitude plot for a window
    window_size = 512
    orig_spec = np.abs(np.fft.rfft(original[:window_size]))
    freq_axis = np.arange(len(orig_spec)) * sample_rate / window_size
    ax3.semilogy(freq_axis, orig_spec + 1e-10, 'b-', linewidth=0.8)
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Magnitude')
    ax3.set_title('Original Spectrum')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, sample_rate / 2])
    
    # Plot 4: Spectrogram of processed
    ax4 = axes[1, 1]
    proc_spec = np.abs(np.fft.rfft(processed[:window_size]))
    ax4.semilogy(freq_axis, proc_spec + 1e-10, 'r-', linewidth=0.8)
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Magnitude')
    ax4.set_title('Processed Spectrum (Alien Artifacts)')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, sample_rate / 2])
    
    plt.tight_layout()
    plt.savefig('spectral_artifact_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("    Visualization saved: spectral_artifact_analysis.png")


def demo_all_effects(input_file: str):
    """Demo all individual effects."""
    print("=" * 70)
    print("Spectral Artifact Generator - Effect Demo")
    print("=" * 70)
    
    sample_rate, audio = load_wav(input_file)
    
    # Use first 2 seconds for demo
    demo_samples = min(int(2 * sample_rate), len(audio))
    audio = audio[:demo_samples]
    
    FRAME_SIZE = 1024
    HOP_SIZE = 512
    
    window = np.hanning(FRAME_SIZE).astype(np.float32)
    
    # Frame
    num_frames = 1 + (len(audio) - FRAME_SIZE) // HOP_SIZE
    frames = []
    for i in range(num_frames):
        start = i * HOP_SIZE
        end = start + FRAME_SIZE
        if end <= len(audio):
            frames.append(audio[start:end])
        else:
            frame = np.zeros(FRAME_SIZE)
            frame[:len(audio) - start] = audio[start:]
            frames.append(frame)
    frames = np.array(frames)
    
    # DCT
    dct_coeffs = apply_dct_variable(frames, window)
    
    effects = [
        ("Original", {}),
        ("Non-Linear Sine", {'nonlinear': True, 'mode': 'sine', 'qstep': 50}),
        ("Spectral Gating", {'gating': True, 'freeze_prob': 0.2, 'gain_range': (0, 15)}),
        ("Spectral Smear", {'smear': True, 'smear_shift': 3}),
    ]
    
    for effect_name, params in effects:
        print(f"\nProcessing: {effect_name}")
        
        processed = []
        for coeff in dct_coeffs:
            p = coeff.copy()
            
            if params.get('nonlinear', False):
                p = nonlinear_quantize(p, params.get('qstep', 10), params.get('mode', 'sine'))
            
            if params.get('gating', False):
                p = spectral_gate_freeze(p, params.get('freeze_prob', 0.1), 
                                         params.get('gain_range', (0, 10)))
            
            if params.get('smear', False):
                p = spectral_smear(p, params.get('smear_shift', 1))
            
            rec = idct(p, type=2, norm='ortho')
            processed.append(rec * window)
        
        processed = np.array(processed)
        
        # Reconstruct
        output_length = (num_frames - 1) * HOP_SIZE + FRAME_SIZE
        output = np.zeros(output_length, dtype=np.float32)
        weights = np.zeros(output_length, dtype=np.float32)
        
        for i in range(num_frames):
            start = i * HOP_SIZE
            end = start + FRAME_SIZE
            output[start:end] += processed[i]
            weights[start:end] += window ** 2
        
        weights[weights == 0] = 1
        output = output / weights
        output = output[:demo_samples]
        
        # Save
        filename = f"effect_{effect_name.replace(' ', '_').lower()}.wav"
        audio_clipped = np.clip(output, -1.0, 1.0)
        audio_int16 = (audio_clipped * 32767).astype(np.int16)
        wavfile.write(filename, sample_rate, audio_int16)
        print(f"    Saved: {filename}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Spectral Artifact Generator')
    parser.add_argument('input', help='Input WAV file')
    parser.add_argument('-o', '--output', default='spectral_morph.wav',
                       help='Output WAV file (default: spectral_morph.wav)')
    parser.add_argument('-d', '--duration', type=float, default=10.0,
                       help='Morph duration in seconds (default: 10)')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo of all individual effects')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_all_effects(args.input)
    else:
        create_morph_effect(args.input, args.output, args.duration)
