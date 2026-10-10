import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

__all__ = [
    'pre_emphasis',
    'de_emphasis',
    'superheterodyne_lo_freq',
    'superheterodyne_tuning_range',
    'image_frequency',
    'agc_gain',
    'plot_pre_de_emphasis'
]

def pre_emphasis(signal_in, fs, tau=75e-6):
    """
    Apply a pre-emphasis filter (first-order high-pass) to the signal.
    
    Parameters:
    - signal_in: Input signal array.
    - fs: Sampling frequency in Hz.
    - tau: Time constant in seconds (default 75us for FM).
    
    Returns:
    - Filtered signal.
    """
    # Create a pole at a high frequency to limit infinite gain
    tau_p = 1 / (2 * np.pi * 100e3)
    num = [tau, 1]
    den = [tau_p, 1]
    b, a = signal.bilinear(num, den, fs=fs)
    return signal.lfilter(b, a, signal_in)

def de_emphasis(signal_in, fs, tau=75e-6):
    """
    Apply a de-emphasis filter (first-order low-pass) to the signal.
    
    Parameters:
    - signal_in: Input signal array.
    - fs: Sampling frequency in Hz.
    - tau: Time constant in seconds (default 75us for FM).
    
    Returns:
    - Filtered signal.
    """
    num = [1]
    den = [tau, 1]
    b, a = signal.bilinear(num, den, fs=fs)
    return signal.lfilter(b, a, signal_in)

def superheterodyne_lo_freq(rf_freq, if_freq, high_side=True):
    """
    Calculate local oscillator (LO) frequency for a superheterodyne receiver.
    
    Parameters:
    - rf_freq: RF frequency.
    - if_freq: Intermediate frequency.
    - high_side: If True, fLO = fRF + fIF. If False, fLO = fRF - fIF.
    
    Returns:
    - LO frequency.
    """
    if high_side:
        return rf_freq + if_freq
    else:
        return rf_freq - if_freq

def superheterodyne_tuning_range(rf_min, rf_max, if_freq, high_side=True):
    """
    Calculate LO frequency range and tuning capacitor ratio.
    
    Parameters:
    - rf_min: Minimum RF frequency.
    - rf_max: Maximum RF frequency.
    - if_freq: Intermediate frequency.
    - high_side: If True, uses high-side injection.
    
    Returns:
    - Dictionary with lo_min, lo_max, and capacitor_ratio.
    """
    lo_min = superheterodyne_lo_freq(rf_min, if_freq, high_side)
    lo_max = superheterodyne_lo_freq(rf_max, if_freq, high_side)
    # Capacitor ratio C_max / C_min = (fLO_max / fLO_min)^2
    c_ratio = (lo_max / lo_min)**2
    return {
        'lo_min': lo_min,
        'lo_max': lo_max,
        'capacitor_ratio': c_ratio
    }

def image_frequency(rf_freq, if_freq, high_side=True):
    """
    Calculate the image frequency in a superheterodyne receiver.
    
    Parameters:
    - rf_freq: RF frequency.
    - if_freq: Intermediate frequency.
    - high_side: If True, LO is above RF, else below.
    
    Returns:
    - Image frequency.
    """
    if high_side:
        return rf_freq + 2 * if_freq
    else:
        return rf_freq - 2 * if_freq

def agc_gain(target_level, input_level, in_db=True):
    """
    Calculate simple AGC (Automatic Gain Control) gain in dB.
    
    Parameters:
    - target_level: Desired signal level (in dB/dBm if in_db=True, or linear scale).
    - input_level: Current input signal level (in dB/dBm if in_db=True, or linear scale).
    - in_db (bool): If True (default), inputs are in dB/dBm so Gain_dB = target - input.
                    If False, inputs are linear so Gain_dB = 20 * log10(target / input).
    
    Returns:
    - AGC gain in dB.
    """
    if in_db:
        return float(target_level - input_level)
    return float(20 * np.log10(target_level / input_level))

def plot_pre_de_emphasis(fs, tau=75e-6):
    """
    Plot the frequency response of pre-emphasis and de-emphasis filters.
    
    Parameters:
    - fs: Sampling frequency.
    - tau: Time constant in seconds.
    
    Returns:
    - matplotlib Figure object.
    """
    tau_p = 1 / (2 * np.pi * 100e3)
    b_pre, a_pre = signal.bilinear([tau, 1], [tau_p, 1], fs=fs)
    b_de, a_de = signal.bilinear([1], [tau, 1], fs=fs)
    
    w_pre, h_pre = signal.freqz(b_pre, a_pre, worN=2048)
    w_de, h_de = signal.freqz(b_de, a_de, worN=2048)
    
    f = w_pre * fs / (2 * np.pi)
    
    # Avoid log of zero
    valid = f > 0
    f = f[valid]
    h_pre = h_pre[valid]
    h_de = h_de[valid]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(f, 20*np.log10(np.abs(h_pre)), label='Pre-emphasis')
    ax.plot(f, 20*np.log10(np.abs(h_de)), label='De-emphasis')
    ax.set_xscale('log')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title(f'FM Pre/De-emphasis Filters (tau={tau*1e6:.0f} µs)')
    ax.grid(True, which="both", ls="-")
    ax.legend()
    return fig
