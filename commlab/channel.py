"""
Channel module for CommSimLab.

This module provides functions for AWGN channel simulation, SNR calculation,
noise figure computation, and cascaded noise calculations using the Friis formula.
"""

import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    'awgn_channel',
    'snr_db',
    'noise_figure_db',
    'cascaded_noise_figure',
    'noise_temperature',
    'plot_signal_with_noise'
]

def awgn_channel(signal, snr_db):
    """
    Add AWGN noise to a signal at a given SNR in dB.
    
    Args:
        signal (ndarray): Input signal array.
        snr_db (float): Target Signal-to-Noise Ratio in dB.
        
    Returns:
        tuple: (noisy_signal, noise) where noisy_signal is the signal with added noise
               and noise is the generated noise array.
    """
    signal = np.asarray(signal)
    
    # Signal power
    if np.iscomplexobj(signal):
        sig_power = np.mean(np.abs(signal)**2)
    else:
        sig_power = np.mean(signal**2)
        
    # Noise power
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    
    # Generate noise
    if np.iscomplexobj(signal):
        noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    else:
        noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
        
    return signal + noise, noise

def snr_db(signal, noise):
    """
    Compute SNR in dB from signal and noise arrays.
    
    Args:
        signal (ndarray): Original signal array.
        noise (ndarray): Noise array.
        
    Returns:
        float: SNR in dB.
    """
    signal = np.asarray(signal)
    noise = np.asarray(noise)
    
    sig_power = np.mean(np.abs(signal)**2) if np.iscomplexobj(signal) else np.mean(signal**2)
    noise_power = np.mean(np.abs(noise)**2) if np.iscomplexobj(noise) else np.mean(noise**2)
    
    if noise_power == 0:
        return np.inf
        
    return 10 * np.log10(sig_power / noise_power)

def noise_figure_db(snr_in_db, snr_out_db):
    """
    Compute noise figure in dB.
    
    Args:
        snr_in_db (float): Input SNR in dB.
        snr_out_db (float): Output SNR in dB.
        
    Returns:
        float: Noise figure in dB.
    """
    return snr_in_db - snr_out_db

def cascaded_noise_figure(gains_db, nf_db_list):
    """
    Friis formula for cascaded amplifier stages.
    
    Args:
        gains_db (list): List of linear gains in dB for each stage.
        nf_db_list (list): List of noise figures in dB for each stage.
        
    Returns:
        float: Overall noise figure in dB.
    """
    if len(gains_db) != len(nf_db_list):
        raise ValueError("Lengths of gains_db and nf_db_list must be equal.")
        
    # Convert from dB to linear scale
    f = [10**(nf/10.0) for nf in nf_db_list]
    g = [10**(g/10.0) for g in gains_db]
    
    f_total = f[0]
    g_cumulative = 1.0
    
    for i in range(1, len(f)):
        g_cumulative *= g[i-1]
        f_total += (f[i] - 1) / g_cumulative
        
    return 10 * np.log10(f_total)

def noise_temperature(nf_db, t0=290):
    """
    Equivalent noise temperature from noise figure.
    
    Args:
        nf_db (float): Noise figure in dB.
        t0 (float, optional): Reference temperature in Kelvin. Defaults to 290.
        
    Returns:
        float: Equivalent noise temperature in Kelvin.
    """
    f = 10**(nf_db / 10.0)
    return t0 * (f - 1)

def plot_signal_with_noise(t, original, noisy, title=''):
    """
    Matplotlib figure showing original and noisy signal.
    
    Args:
        t (ndarray): Time array.
        original (ndarray): Original signal array.
        noisy (ndarray): Noisy signal array.
        title (str, optional): Plot title. Defaults to ''.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, noisy, label='Noisy Signal', alpha=0.7, color='lightcoral')
    ax.plot(t, original, label='Original Signal', linewidth=2, color='darkblue')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.set_title(title if title else 'Signal with AWGN')
    ax.legend()
    ax.grid(True)
    return fig
