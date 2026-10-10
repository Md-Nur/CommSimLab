import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square

__all__ = [
    'nyquist_rate',
    'sample_signal',
    'natural_sampling',
    'flat_top_sampling',
    'reconstruct_signal',
    'check_aliasing',
    'plot_sampling_demo',
    'plot_aliasing_demo'
]

def nyquist_rate(max_freq):
    """
    Return the minimum sampling rate (Nyquist rate) for a given maximum frequency.
    
    Args:
        max_freq (float): Maximum frequency of the signal in Hz.
        
    Returns:
        float: Minimum sampling rate (2 * fmax).
    """
    return 2.0 * max_freq

def sample_signal(signal, t, fs_original, fs_sample):
    """
    Sample a signal at a specified sampling rate.
    
    Args:
        signal (ndarray): Original continuous-time signal (highly sampled).
        t (ndarray): Time vector for the original signal.
        fs_original (float): Original sampling rate.
        fs_sample (float): Desired sampling rate.
        
    Returns:
        tuple: (sample_times, samples, sample_indices)
    """
    step = int(np.round(fs_original / fs_sample))
    if step < 1:
        step = 1
    sample_indices = np.arange(0, len(t), step)
    return t[sample_indices], signal[sample_indices], sample_indices

def natural_sampling(signal, t, fs_sample, pulse_width_fraction=0.1):
    """
    Perform natural (gating) sampling on a signal.
    
    Args:
        signal (ndarray): Original continuous-time signal.
        t (ndarray): Time vector.
        fs_sample (float): Sampling frequency.
        pulse_width_fraction (float): Fraction of the sampling period the pulse is high.
        
    Returns:
        tuple: (sampled_signal, pulse_train)
    """
    ts = 1.0 / fs_sample
    duty = pulse_width_fraction
    # Generate pulse train (0 or 1)
    pulse_train = 0.5 * (1 + square(2 * np.pi * fs_sample * t, duty=duty))
    sampled_signal = signal * pulse_train
    return sampled_signal, pulse_train

def flat_top_sampling(signal, t, fs_sample, pulse_width_fraction=0.1):
    """
    Perform flat-top (sample-and-hold) sampling on a signal.
    
    Args:
        signal (ndarray): Original continuous-time signal.
        t (ndarray): Time vector.
        fs_sample (float): Sampling frequency.
        pulse_width_fraction (float): Fraction of the sampling period the pulse is high.
        
    Returns:
        ndarray: Flat-top sampled signal.
    """
    ts = 1.0 / fs_sample
    sampled_signal = np.zeros_like(signal)
    pulse_width_time = pulse_width_fraction * ts
    
    # Simple discrete implementation
    for i, t_val in enumerate(t):
        n = int(np.floor(t_val / ts))
        t_start = n * ts
        if t_start <= t_val < t_start + pulse_width_time:
            # Find closest index for t_start
            idx_start = np.argmin(np.abs(t - t_start))
            sampled_signal[i] = signal[idx_start]
            
    return sampled_signal

def reconstruct_signal(samples, sample_times, t_reconstruct, fs_sample):
    """
    Reconstruct a continuous-time signal from samples using sinc interpolation.
    
    Args:
        samples (ndarray): The sampled signal values.
        sample_times (ndarray): The times at which the samples were taken.
        t_reconstruct (ndarray): The time vector to reconstruct the signal on.
        fs_sample (float): The sampling frequency.
        
    Returns:
        ndarray: Reconstructed continuous-time signal.
    """
    ts = 1.0 / fs_sample
    reconstructed = np.zeros_like(t_reconstruct, dtype=float)
    for sample, t_n in zip(samples, sample_times):
        reconstructed += sample * np.sinc((t_reconstruct - t_n) / ts)
    return reconstructed

def check_aliasing(signal_freq, fs_sample):
    """
    Check if aliasing occurs for a given signal frequency and sampling rate.
    
    Args:
        signal_freq (float): Frequency of the signal in Hz.
        fs_sample (float): Sampling frequency in Hz.
        
    Returns:
        tuple: (is_aliasing (bool), aliased_freq (float))
    """
    nyq = fs_sample / 2.0
    # The aliased frequency wraps around the Nyquist frequency
    # We find the fundamental alias in the principal Nyquist zone [0, fs/2]
    num_folds = int(np.floor(signal_freq / nyq))
    rem = signal_freq % nyq
    if num_folds % 2 == 0:
        aliased_freq = rem
    else:
        aliased_freq = nyq - rem
        
    is_aliasing = (signal_freq > nyq)
    return is_aliasing, aliased_freq

def plot_sampling_demo(t, original, sampled=None, reconstructed=None, title='Sampling Demonstration'):
    """
    Plot a multi-panel demonstration of sampling and reconstruction.
    
    Args:
        t (ndarray): Time vector.
        original (ndarray): Original continuous-time signal.
        sampled (ndarray or tuple, optional): Sampled signal, or (sample_times, samples) tuple.
        reconstructed (ndarray, optional): Reconstructed signal.
        title (str): Title for the figure.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    panels = ['original']
    if sampled is not None:
        panels.append('sampled')
    if reconstructed is not None:
        panels.append('reconstructed')
        
    num_plots = len(panels)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots), sharex=True)
    if num_plots == 1:
        axes = [axes]
        
    for i, panel in enumerate(panels):
        ax = axes[i]
        if panel == 'original':
            ax.plot(t, original, 'b-')
            ax.set_title('Original Signal')
            ax.set_ylabel('Amplitude')
            ax.grid(True)
        elif panel == 'sampled':
            if isinstance(sampled, (tuple, list)) and len(sampled) == 2:
                s_times, s_vals = sampled
                ax.stem(s_times, s_vals, linefmt='r-', markerfmt='ro', basefmt='k-')
            elif hasattr(sampled, '__len__') and len(sampled) != len(t):
                s_times = np.linspace(t[0], t[-1], len(sampled))
                ax.stem(s_times, sampled, linefmt='r-', markerfmt='ro', basefmt='k-')
            else:
                ax.plot(t, sampled, 'r-')
            ax.set_title('Sampled Signal')
            ax.set_ylabel('Amplitude')
            ax.grid(True)
        elif panel == 'reconstructed':
            ax.plot(t, reconstructed, 'g-')
            ax.set_title('Reconstructed Signal')
            ax.set_ylabel('Amplitude')
            ax.grid(True)
            
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_aliasing_demo(signal_freq, fs_list):
    """
    Demonstrate aliasing by plotting a sine wave sampled at different rates.
    
    Args:
        signal_freq (float): Frequency of the original signal in Hz.
        fs_list (list): List of sampling frequencies to demonstrate.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    duration = 5.0 / signal_freq
    t_fine = np.linspace(0, duration, 1000)
    original_signal = np.cos(2 * np.pi * signal_freq * t_fine)
    
    num_plots = len(fs_list)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots), sharex=True)
    if num_plots == 1:
        axes = [axes]
        
    for ax, fs in zip(axes, fs_list):
        t_sample = np.arange(0, duration, 1.0 / fs)
        samples = np.cos(2 * np.pi * signal_freq * t_sample)
        
        # Reconstruct
        reconstructed = reconstruct_signal(samples, t_sample, t_fine, fs)
        
        ax.plot(t_fine, original_signal, 'k--', alpha=0.5, label='Original')
        ax.plot(t_fine, reconstructed, 'r-', label='Reconstructed')
        ax.stem(t_sample, samples, linefmt='b-', markerfmt='bo', basefmt='b-', label='Samples')
        
        is_aliased, alias_f = check_aliasing(signal_freq, fs)
        title = f'fs = {fs} Hz '
        if is_aliased:
            title += f'(Aliased to {alias_f:.2f} Hz)'
        else:
            title += '(No Aliasing)'
            
        ax.set_title(title)
        ax.set_ylabel('Amplitude')
        ax.grid(True)
        ax.legend(loc='upper right')
        
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'Aliasing Demo (Signal Freq = {signal_freq} Hz)')
    plt.tight_layout()
    return fig
