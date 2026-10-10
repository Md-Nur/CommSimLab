"""
Pulse Code Modulation (PCM) module.
Implements sampling, quantization, encoding, decoding, and reconstruction for PCM.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import from other commlab modules as requested
from commlab.sampling import sample_signal, reconstruct_signal
from commlab.quantization import uniform_quantize

__all__ = [
    'pcm_encode',
    'pcm_decode',
    'pcm_reconstruct',
    'pcm_bitrate',
    'pcm_bandwidth',
    'plot_pcm_chain'
]

def pcm_encode(signal, t, fs_sample, num_bits, v_min=None, v_max=None):
    """
    Full PCM encoding pipeline: sample -> quantize -> binary encode.
    
    Args:
        signal (ndarray): Original analog signal array.
        t (ndarray): Time array for the signal.
        fs_sample (float): Sampling frequency in Hz.
        num_bits (int): Number of bits per sample.
        v_min (float, optional): Minimum voltage for quantization.
        v_max (float, optional): Maximum voltage for quantization.
        
    Returns:
        dict: A dictionary containing:
            'samples': Array of sampled signal values.
            'sample_times': Array of sample times.
            'quantized': Array of quantized sample values.
            'codes': List of binary string codes representing the quantized levels.
            'step_size': Quantization step size.
            'levels': Array of quantization levels.
            'v_min': The v_min used in quantization.
            'num_bits': Number of bits used.
    """
    # 1. Sampling
    if len(t) > 1:
        fs_original = 1.0 / (t[1] - t[0])
    else:
        fs_original = fs_sample
    sample_times, samples, _ = sample_signal(signal, t, fs_original, fs_sample)
    
    if v_min is None:
        v_min = float(np.min(samples))
    if v_max is None:
        v_max = float(np.max(samples))
        
    # 2. Quantization
    quantized, levels, step_size = uniform_quantize(samples, num_bits, v_min, v_max)
    
    # 3. Binary Encoding
    num_levels = 2 ** num_bits
    clipped = np.clip(samples, v_min, v_max)
    indices = np.floor((clipped - v_min) / step_size).astype(int)
    indices = np.clip(indices, 0, num_levels - 1)
    codes = [format(int(idx), f'0{num_bits}b') for idx in indices]
    
    return {
        'samples': samples,
        'sample_times': sample_times,
        'quantized': quantized,
        'codes': codes,
        'step_size': step_size,
        'levels': levels,
        'v_min': v_min,
        'num_bits': num_bits
    }

def pcm_decode(codes, step_size, v_min, num_bits):
    """
    Decode PCM binary codes back to quantized values.
    
    Args:
        codes (list): List of binary strings.
        step_size (float): Quantization step size.
        v_min (float): Minimum voltage level used during encoding.
        num_bits (int): Number of bits per sample.
        
    Returns:
        ndarray: Array of decoded (quantized) sample values.
    """
    indices = np.array([int(code, 2) for code in codes])
    # Map index back to the mid-point of the quantization level
    decoded_samples = v_min + (indices * step_size) + (step_size / 2.0)
    return decoded_samples

def pcm_reconstruct(decoded_samples, sample_times, t_reconstruct, fs_sample):
    """
    Reconstruct analog signal from decoded PCM using sinc interpolation.
    
    Args:
        decoded_samples (ndarray): The decoded sample values.
        sample_times (ndarray): The times at which the samples were taken.
        t_reconstruct (ndarray): The continuous time array for reconstruction.
        fs_sample (float): The sampling frequency.
        
    Returns:
        ndarray: The reconstructed continuous-time signal.
    """
    return reconstruct_signal(decoded_samples, sample_times, t_reconstruct, fs_sample)

def pcm_bitrate(fs_sample, num_bits):
    """
    Calculate PCM bit rate.
    
    Args:
        fs_sample (float): Sampling frequency in Hz.
        num_bits (int): Number of bits per sample.
        
    Returns:
        float: Bit rate in bits per second (bps).
    """
    return fs_sample * num_bits

def pcm_bandwidth(fs_sample, num_bits):
    """
    Calculate minimum PCM bandwidth.
    
    Args:
        fs_sample (float): Sampling frequency in Hz.
        num_bits (int): Number of bits per sample.
        
    Returns:
        float: Minimum transmission bandwidth in Hz.
    """
    return pcm_bitrate(fs_sample, num_bits) / 2.0

def plot_pcm_chain(t, original, sample_times, samples, quantized_samples, reconstructed, title='PCM Chain'):
    """
    Plot a multi-panel figure showing the entire PCM process.
    
    Args:
        t (ndarray): Original time array.
        original (ndarray): Original analog signal.
        sample_times (ndarray): Array of sample times.
        samples (ndarray): Sampled values.
        quantized_samples (ndarray): Quantized sample values.
        reconstructed (ndarray): Reconstructed analog signal.
        title (str): Title for the plot.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # Original + Sampling
    axes[0].plot(t, original, label='Original Analog', color='blue')
    axes[0].stem(sample_times, samples, linefmt='r-', markerfmt='ro', basefmt='r-', label='Sampled')
    axes[0].set_title(title + " - Sampling")
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    axes[0].grid(True)
    
    # Quantized
    axes[1].plot(t, original, label='Original Analog', color='blue', alpha=0.3)
    axes[1].step(sample_times, quantized_samples, where='mid', label='Quantized', color='orange')
    axes[1].plot(sample_times, quantized_samples, 'o', color='orange', markersize=4)
    axes[1].set_title("Quantization")
    axes[1].set_ylabel('Amplitude')
    axes[1].legend()
    axes[1].grid(True)
    
    # Reconstructed
    axes[2].plot(t, original, label='Original Analog', color='blue', alpha=0.5)
    axes[2].plot(t, reconstructed, label='Reconstructed', color='green')
    axes[2].set_title("Reconstruction (Decoding & Smoothing)")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel('Amplitude')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    return fig
