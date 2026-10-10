"""
Capacity module for CommSimLab.

This module provides functions for calculating Shannon capacity, Nyquist maximum
data rate, and related parameters for communication channels.
"""

import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    'shannon_capacity',
    'shannon_capacity_db',
    'nyquist_max_rate',
    'min_nyquist_levels',
    'bandwidth_for_capacity',
    'plot_capacity_vs_snr',
    'plot_capacity_vs_bandwidth'
]

def shannon_capacity(bandwidth_hz, snr_linear):
    """
    Calculate Shannon capacity from linear SNR.
    
    C = B * log2(1 + SNR)
    
    Args:
        bandwidth_hz (float): Channel bandwidth in Hz.
        snr_linear (float): Linear signal-to-noise ratio.
        
    Returns:
        float: Shannon capacity in bits per second (bps).
    """
    return bandwidth_hz * np.log2(1 + snr_linear)

def shannon_capacity_db(bandwidth_hz, snr_db):
    """
    Calculate Shannon capacity from SNR in dB.
    
    Args:
        bandwidth_hz (float): Channel bandwidth in Hz.
        snr_db (float): Signal-to-noise ratio in dB.
        
    Returns:
        float: Shannon capacity in bits per second (bps).
    """
    snr_linear = 10 ** (snr_db / 10.0)
    return shannon_capacity(bandwidth_hz, snr_linear)

def nyquist_max_rate(bandwidth_hz, levels):
    """
    Calculate Nyquist maximum data rate for a noiseless channel.
    
    R = 2B * log2(M)
    
    Args:
        bandwidth_hz (float): Channel bandwidth in Hz.
        levels (int): Number of signal levels (M).
        
    Returns:
        float: Maximum data rate in bits per second (bps).
    """
    return 2 * bandwidth_hz * np.log2(levels)

def min_nyquist_levels(bandwidth_hz, data_rate_bps):
    """
    Calculate minimum number of signal levels to achieve a given data rate.
    
    Args:
        bandwidth_hz (float): Channel bandwidth in Hz.
        data_rate_bps (float): Target data rate in bps.
        
    Returns:
        int: Minimum number of levels M (integer, ceiling).
    """
    # data_rate = 2 * B * log2(M) => log2(M) = data_rate / (2 * B) => M = 2^(data_rate / (2 * B))
    m = 2 ** (data_rate_bps / (2 * bandwidth_hz))
    return int(np.ceil(m))

def bandwidth_for_capacity(capacity_bps, snr_db):
    """
    Calculate minimum bandwidth needed for a given capacity and SNR.
    
    Args:
        capacity_bps (float): Target capacity in bps.
        snr_db (float): Signal-to-noise ratio in dB.
        
    Returns:
        float: Minimum bandwidth in Hz.
    """
    snr_linear = 10 ** (snr_db / 10.0)
    # C = B * log2(1 + SNR) => B = C / log2(1 + SNR)
    return capacity_bps / np.log2(1 + snr_linear)

def plot_capacity_vs_snr(bandwidth_hz, snr_db_range):
    """
    Plot Shannon capacity vs SNR.
    
    Args:
        bandwidth_hz (float): Channel bandwidth in Hz.
        snr_db_range (array-like): Range of SNR values in dB.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    snr_db_range = np.asarray(snr_db_range)
    capacity = [shannon_capacity_db(bandwidth_hz, snr) for snr in snr_db_range]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(snr_db_range, capacity, linewidth=2, color='blue')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Capacity (bps)')
    ax.set_title(f'Shannon Capacity vs SNR (Bandwidth = {bandwidth_hz} Hz)')
    ax.grid(True)
    
    return fig

def plot_capacity_vs_bandwidth(snr_db, bandwidth_range):
    """
    Plot Shannon capacity vs bandwidth.
    
    Args:
        snr_db (float): Signal-to-noise ratio in dB.
        bandwidth_range (array-like): Range of bandwidth values in Hz.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    bandwidth_range = np.asarray(bandwidth_range)
    capacity = [shannon_capacity_db(bw, snr_db) for bw in bandwidth_range]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(bandwidth_range, capacity, linewidth=2, color='green')
    ax.set_xlabel('Bandwidth (Hz)')
    ax.set_ylabel('Capacity (bps)')
    ax.set_title(f'Shannon Capacity vs Bandwidth (SNR = {snr_db} dB)')
    ax.grid(True)
    
    return fig
