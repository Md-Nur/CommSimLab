"""
Digital line coding schemes and their Power Spectral Densities.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

__all__ = [
    'nrz_unipolar',
    'nrz_polar',
    'nrz_inverted',
    'rz_unipolar',
    'rz_polar',
    'rz_ami',
    'manchester',
    'diff_manchester',
    'plot_line_code',
    'plot_all_line_codes',
    'plot_psd',
    'plot_all_psd',
    'random_bits'
]

def random_bits(n: int) -> np.ndarray:
    """Generate n random binary bits."""
    return np.random.randint(0, 2, n)

def _get_time_vector(num_bits: int, bit_rate: float, fs: float) -> tuple[np.ndarray, int]:
    """Calculate time vector and samples per bit."""
    duration = num_bits / bit_rate
    samples_per_bit = int(round(fs / bit_rate))
    
    total_samples = samples_per_bit * num_bits
    t = np.arange(total_samples) / fs
    return t, samples_per_bit

def nrz_unipolar(bits: np.ndarray, bit_rate: float, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """NRZ-L unipolar: 1->high, 0->zero"""
    t, spb = _get_time_vector(len(bits), bit_rate, fs)
    signal = np.repeat(bits, spb)
    return t, signal.astype(float)

def nrz_polar(bits: np.ndarray, bit_rate: float, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """NRZ-L polar: 1->+V, 0->-V"""
    t, spb = _get_time_vector(len(bits), bit_rate, fs)
    signal = np.repeat(bits, spb)
    signal = 2.0 * signal - 1.0
    return t, signal

def nrz_inverted(bits: np.ndarray, bit_rate: float, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """NRZ-I: transition on 1, no transition on 0"""
    t, spb = _get_time_vector(len(bits), bit_rate, fs)
    levels = np.zeros(len(bits))
    current_level = 1.0
    for i, b in enumerate(bits):
        if b == 1:
            current_level *= -1
        levels[i] = current_level
    signal = np.repeat(levels, spb)
    return t, signal

def rz_unipolar(bits: np.ndarray, bit_rate: float, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """RZ unipolar: 1->pulse for half bit period, 0->zero"""
    t, spb = _get_time_vector(len(bits), bit_rate, fs)
    half_spb = spb // 2
    signal = np.zeros(len(bits) * spb)
    for i, b in enumerate(bits):
        if b == 1:
            signal[i * spb : i * spb + half_spb] = 1.0
    return t, signal

def rz_polar(bits: np.ndarray, bit_rate: float, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """RZ polar: 1->+V for half, 0->-V for half, return to zero"""
    t, spb = _get_time_vector(len(bits), bit_rate, fs)
    half_spb = spb // 2
    signal = np.zeros(len(bits) * spb)
    for i, b in enumerate(bits):
        if b == 1:
            signal[i * spb : i * spb + half_spb] = 1.0
        else:
            signal[i * spb : i * spb + half_spb] = -1.0
    return t, signal

def rz_ami(bits: np.ndarray, bit_rate: float, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """RZ-AMI (bipolar): alternate +/-V for 1s, 0->zero"""
    t, spb = _get_time_vector(len(bits), bit_rate, fs)
    half_spb = spb // 2
    signal = np.zeros(len(bits) * spb)
    last_pulse = -1.0
    for i, b in enumerate(bits):
        if b == 1:
            last_pulse *= -1
            signal[i * spb : i * spb + half_spb] = last_pulse
    return t, signal

def manchester(bits: np.ndarray, bit_rate: float, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Manchester (IEEE 802.3): 1->high-to-low transition, 0->low-to-high at mid-bit"""
    t, spb = _get_time_vector(len(bits), bit_rate, fs)
    half_spb = spb // 2
    signal = np.zeros(len(bits) * spb)
    for i, b in enumerate(bits):
        if b == 1:
            signal[i * spb : i * spb + half_spb] = 1.0
            signal[i * spb + half_spb : (i+1) * spb] = -1.0
        else:
            signal[i * spb : i * spb + half_spb] = -1.0
            signal[i * spb + half_spb : (i+1) * spb] = 1.0
    return t, signal

def diff_manchester(bits: np.ndarray, bit_rate: float, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Differential Manchester: always mid-bit transition, 0->transition at start, 1->no transition at start"""
    t, spb = _get_time_vector(len(bits), bit_rate, fs)
    half_spb = spb // 2
    signal = np.zeros(len(bits) * spb)
    
    current_level = 1.0
    for i, b in enumerate(bits):
        if b == 0:
            current_level *= -1
        
        signal[i * spb : i * spb + half_spb] = current_level
        current_level *= -1
        signal[i * spb + half_spb : (i+1) * spb] = current_level
        
    return t, signal

def plot_line_code(t: np.ndarray, signal: np.ndarray, bits: np.ndarray, bit_rate: float, title: str = '') -> plt.Figure:
    """Plot encoded signal with bit boundaries and labels."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.step(t, signal, where='post', linewidth=2)
    
    bit_duration = 1.0 / bit_rate
    num_bits = len(bits)
    
    for i in range(num_bits + 1):
        ax.axvline(i * bit_duration, color='gray', linestyle='--', alpha=0.5)
        
    y_max = max(max(signal) * 1.2 if np.max(signal) > 0 else 1.2, 1.2)
    for i, b in enumerate(bits):
        ax.text(i * bit_duration + bit_duration / 2, y_max, str(b), 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    ax.set_ylim([min(np.min(signal)-0.5, -1.5), max(np.max(signal)+0.5, 1.5) + 0.5])
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

def plot_all_line_codes(bits: np.ndarray, bit_rate: float, fs: float) -> plt.Figure:
    """All encoding schemes side by side for the same bit pattern."""
    funcs = [
        ('NRZ Unipolar', nrz_unipolar),
        ('NRZ Polar', nrz_polar),
        ('NRZ Inverted', nrz_inverted),
        ('RZ Unipolar', rz_unipolar),
        ('RZ Polar', rz_polar),
        ('RZ AMI', rz_ami),
        ('Manchester', manchester),
        ('Differential Manchester', diff_manchester)
    ]
    
    fig, axes = plt.subplots(len(funcs), 1, figsize=(10, 1.5 * len(funcs)), sharex=True)
    bit_duration = 1.0 / bit_rate
    
    for ax, (name, func) in zip(axes, funcs):
        t, signal = func(bits, bit_rate, fs)
        ax.step(t, signal, where='post', linewidth=1.5)
        for i in range(len(bits) + 1):
            ax.axvline(i * bit_duration, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(name, loc='left', fontsize=10)
        ax.set_yticks([-1, 0, 1])
        ax.set_ylim([-1.5, 1.5])
        ax.grid(True, alpha=0.3)
        
        if ax == axes[0]:
            for i, b in enumerate(bits):
                ax.text(i * bit_duration + bit_duration / 2, 1.6, str(b), 
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()
    return fig

def plot_psd(signal: np.ndarray, fs: float, title: str = 'Power Spectral Density') -> plt.Figure:
    """Plot PSD using Welch's method."""
    nperseg = min(len(signal), 1024)
    f, Pxx = welch(signal, fs, nperseg=nperseg)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(f, 10 * np.log10(np.maximum(Pxx, 1e-12)))
    ax.set_title(title)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density (dB/Hz)')
    ax.grid(True)
    fig.tight_layout()
    return fig

def plot_all_psd(bits: np.ndarray, bit_rate: float, fs: float) -> plt.Figure:
    """PSD for all schemes overlaid or in subplots."""
    funcs = [
        ('NRZ Unipolar', nrz_unipolar),
        ('NRZ Polar', nrz_polar),
        ('NRZ Inverted', nrz_inverted),
        ('RZ Unipolar', rz_unipolar),
        ('RZ Polar', rz_polar),
        ('RZ AMI', rz_ami),
        ('Manchester', manchester),
        ('Differential Manchester', diff_manchester)
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, func in funcs:
        t, signal = func(bits, bit_rate, fs)
        nperseg = min(len(signal), 1024)
        f, Pxx = welch(signal, fs, nperseg=nperseg)
        ax.plot(f, 10 * np.log10(np.maximum(Pxx, 1e-12)), label=name)
        
    ax.set_title('Power Spectral Density Comparison')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (dB/Hz)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    ax.set_xlim([0, bit_rate * 3])
    fig.tight_layout()
    return fig
