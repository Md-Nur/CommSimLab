import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    'uniform_quantize',
    'quantization_noise_power',
    'sqnr_db',
    'sqnr_actual',
    'pam_signal',
    'pam_demodulate',
    'plot_quantization',
    'plot_sqnr_vs_bits'
]

def uniform_quantize(signal, num_bits, v_min=None, v_max=None):
    """
    Uniform mid-rise quantizer.
    
    Args:
        signal (ndarray): Input signal.
        num_bits (int): Number of quantization bits.
        v_min (float, optional): Minimum voltage level. Defaults to min(signal).
        v_max (float, optional): Maximum voltage level. Defaults to max(signal).
        
    Returns:
        tuple: (quantized_signal, quantization_levels, step_size)
    """
    if v_min is None:
        v_min = np.min(signal)
    if v_max is None:
        v_max = np.max(signal)
        
    num_levels = 2 ** num_bits
    step_size = (v_max - v_min) / num_levels
    
    # Clip signal to [v_min, v_max]
    clipped_signal = np.clip(signal, v_min, v_max)
    
    # Scale to indices [0, num_levels-1]
    # For mid-rise quantizer, the levels are symmetrically distributed around zero if v_min = -v_max
    indices = np.floor((clipped_signal - v_min) / step_size).astype(int)
    
    # Fix the highest value which could be at num_levels
    indices[indices >= num_levels] = num_levels - 1
    
    # Quantized values (mid-rise means the values are midpoints of intervals)
    quantization_levels = v_min + step_size * (np.arange(num_levels) + 0.5)
    quantized_signal = quantization_levels[indices]
    
    return quantized_signal, quantization_levels, step_size

def quantization_noise_power(step_size):
    """
    Calculate theoretical quantization noise power.
    
    Args:
        step_size (float): Quantization step size.
        
    Returns:
        float: Noise power (sigma^2 = Delta^2 / 12).
    """
    return (step_size ** 2) / 12.0

def sqnr_db(num_bits):
    """
    Calculate theoretical SQNR for uniform quantizer with sinusoidal input.
    
    Args:
        num_bits (int): Number of quantization bits.
        
    Returns:
        float: SQNR in dB.
    """
    return 6.02 * num_bits + 1.76

def sqnr_actual(signal, quantized_signal):
    """
    Compute actual Signal-to-Quantization-Noise Ratio (SQNR) in dB.
    
    Args:
        signal (ndarray): Original signal.
        quantized_signal (ndarray): Quantized signal.
        
    Returns:
        float: SQNR in dB.
    """
    signal_power = np.mean(signal ** 2)
    noise = signal - quantized_signal
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
        
    return 10 * np.log10(signal_power / noise_power)

def _raised_cosine(t, T, alpha):
    """Helper for raised cosine pulse"""
    # Handle singularity at t = T/(2*alpha)
    np.seterr(divide='ignore', invalid='ignore')
    val = np.sinc(t / T) * np.cos(np.pi * alpha * t / T) / (1 - (2 * alpha * t / T)**2)
    
    # Fix singularities
    if alpha > 0:
        singularity = T / (2 * alpha)
        mask = np.isclose(np.abs(t), singularity)
        val[mask] = (alpha / 2) * np.sin(np.pi / (2 * alpha))
    return val

def pam_signal(bits, bit_rate, fs, pulse_shape='rect', levels=2):
    """
    Generate PAM signal from a bit sequence.
    
    Args:
        bits (list or ndarray): Binary sequence (0s and 1s).
        bit_rate (float): Bit rate in bps.
        fs (float): Sampling frequency.
        pulse_shape (str): 'rect' for rectangular pulse, 'rc' for raised-cosine.
        levels (int): Number of PAM levels (e.g., 2, 4, 8).
        
    Returns:
        tuple: (t, pam_signal)
    """
    import math
    bits = np.array(bits)
    bits_per_symbol = int(math.log2(levels))
    
    # Pad bits if necessary
    if len(bits) % bits_per_symbol != 0:
        padding = bits_per_symbol - (len(bits) % bits_per_symbol)
        bits = np.append(bits, np.zeros(padding, dtype=int))
        
    # Group bits to symbols
    num_symbols = len(bits) // bits_per_symbol
    symbols = np.zeros(num_symbols)
    
    for i in range(num_symbols):
        symbol_bits = bits[i * bits_per_symbol : (i + 1) * bits_per_symbol]
        # Convert binary to integer
        val = 0
        for b in symbol_bits:
            val = (val << 1) | b
        # Map to PAM levels: for L levels, we use -(L-1), -(L-3), ..., (L-3), (L-1)
        symbols[i] = 2 * val - (levels - 1)
        
    T_symbol = bits_per_symbol / bit_rate
    samples_per_symbol = int(fs * T_symbol)
    
    total_duration = num_symbols * T_symbol
    t = np.arange(0, total_duration, 1/fs)
    pam_sig = np.zeros_like(t)
    
    if pulse_shape == 'rect':
        for i, sym in enumerate(symbols):
            start_idx = i * samples_per_symbol
            end_idx = min((i + 1) * samples_per_symbol, len(t))
            pam_sig[start_idx:end_idx] = sym
    elif pulse_shape == 'rc':
        alpha = 0.5 # Default roll-off
        for i, sym in enumerate(symbols):
            t_shifted = t - (i + 0.5) * T_symbol
            pam_sig += sym * _raised_cosine(t_shifted, T_symbol, alpha)
    else:
        raise ValueError("pulse_shape must be 'rect' or 'rc'")
        
    return t, pam_sig

def pam_demodulate(pam_signal, bit_rate, fs, levels=2):
    """
    Demodulate a baseband PAM signal back to bits.
    Assumes rectangular pulse shaping and perfect synchronization.
    
    Args:
        pam_signal (ndarray): Received PAM signal.
        bit_rate (float): Bit rate in bps.
        fs (float): Sampling frequency.
        levels (int): Number of PAM levels.
        
    Returns:
        ndarray: Demodulated bits.
    """
    import math
    bits_per_symbol = int(math.log2(levels))
    T_symbol = bits_per_symbol / bit_rate
    samples_per_symbol = int(fs * T_symbol)
    
    num_symbols = len(pam_signal) // samples_per_symbol
    demodulated_bits = []
    
    # Ideal levels mapping
    # For L levels, values are 2*v - (L-1) where v is in [0, L-1]
    # To get v from sym, v = (sym + L - 1) / 2
    
    for i in range(num_symbols):
        # Sample at center of symbol period
        sample_idx = int((i + 0.5) * samples_per_symbol)
        if sample_idx >= len(pam_signal):
            break
        sym_val = pam_signal[sample_idx]
        
        # Decide which level is closest
        # Valid symbols are -(L-1), ..., (L-1)
        valid_symbols = np.arange(-(levels-1), levels, 2)
        closest_sym = valid_symbols[np.argmin(np.abs(valid_symbols - sym_val))]
        
        v = int((closest_sym + levels - 1) // 2)
        
        # Convert integer to bits
        sym_bits = []
        for _ in range(bits_per_symbol):
            sym_bits.append(v & 1)
            v >>= 1
        demodulated_bits.extend(reversed(sym_bits))
        
    return np.array(demodulated_bits)

def plot_quantization(t, original, quantized, levels, title='Quantization'):
    """
    Plot original and quantized signal with quantization levels.
    
    Args:
        t (ndarray): Time vector.
        original (ndarray): Original continuous-time signal.
        quantized (ndarray): Quantized signal.
        levels (ndarray): Array of quantization levels.
        title (str): Figure title.
        
    Returns:
        matplotlib.figure.Figure: Generated figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, original, 'b-', label='Original Signal', alpha=0.7)
    ax.step(t, quantized, 'r-', label='Quantized Signal', where='post')
    
    # Plot quantization levels as horizontal lines
    for level in levels:
        ax.axhline(y=level, color='gray', linestyle=':', alpha=0.5)
        
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(True)
    ax.legend(loc='upper right')
    plt.tight_layout()
    return fig

def plot_sqnr_vs_bits(max_bits=16):
    """
    Plot theoretical SQNR vs number of bits for a uniform quantizer.
    
    Args:
        max_bits (int): Maximum number of bits to plot.
        
    Returns:
        matplotlib.figure.Figure: Generated figure.
    """
    bits_range = np.arange(1, max_bits + 1)
    sqnr_vals = [sqnr_db(b) for b in bits_range]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bits_range, sqnr_vals, 'bo-')
    ax.set_title('Theoretical SQNR vs Number of Bits')
    ax.set_xlabel('Number of Bits (n)')
    ax.set_ylabel('SQNR (dB)')
    ax.grid(True)
    ax.set_xticks(bits_range)
    plt.tight_layout()
    return fig
