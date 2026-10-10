import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

__all__ = [
    'am_modulate',
    'am_power',
    'am_multi_tone_power',
    'am_efficiency',
    'am_demodulate_envelope',
    'check_overmodulation',
    'fm_modulate',
    'fm_bandwidth_carson',
    'fm_modulation_index',
    'fm_demodulate_discriminator',
    'vsb_filter',
    'plot_am',
    'plot_fm',
    'plot_spectrum',
    'generate_tone',
    'generate_multi_tone'
]

def generate_tone(freq, duration, fs, amplitude=1.0):
    """
    Generate a simple cosine tone.

    Parameters:
    - freq (float): Tone frequency in Hz.
    - duration (float): Signal duration in seconds.
    - fs (float): Sampling frequency in Hz.
    - amplitude (float): Tone amplitude.

    Returns:
    - t (ndarray): Time vector.
    - tone (ndarray): The generated cosine signal.
    """
    t = np.arange(0, duration, 1/fs)
    tone = amplitude * np.cos(2 * np.pi * freq * t)
    return t, tone

def generate_multi_tone(freqs, duration, fs, amplitudes=None):
    """
    Generate a sum of cosine tones.

    Parameters:
    - freqs (list or ndarray): Frequencies of the tones in Hz.
    - duration (float): Signal duration in seconds.
    - fs (float): Sampling frequency in Hz.
    - amplitudes (list or ndarray, optional): Amplitudes of the tones. Defaults to 1.0 for all.

    Returns:
    - t (ndarray): Time vector.
    - multi_tone (ndarray): The sum of cosine signals.
    """
    if amplitudes is None:
        amplitudes = [1.0] * len(freqs)
    
    t = np.arange(0, duration, 1/fs)
    multi_tone = np.zeros_like(t)
    for f, A in zip(freqs, amplitudes):
        multi_tone += A * np.cos(2 * np.pi * f * t)
        
    return t, multi_tone

def am_modulate(message, carrier_freq, fs, modulation_index, carrier_amp=1.0):
    """
    Generate an AM (Amplitude Modulation) signal.

    Parameters:
    - message (ndarray): The modulating message signal (normalized between -1 and 1).
    - carrier_freq (float): Carrier frequency in Hz.
    - fs (float): Sampling frequency in Hz.
    - modulation_index (float): Modulation index (m).
    - carrier_amp (float): Carrier amplitude (Ac).

    Returns:
    - t (ndarray): Time vector.
    - am_signal (ndarray): The modulated AM signal.
    - carrier (ndarray): The unmodulated carrier signal.
    - envelope (ndarray): The envelope of the AM signal.
    """
    t = np.arange(len(message)) / fs
    carrier = carrier_amp * np.cos(2 * np.pi * carrier_freq * t)
    envelope = carrier_amp * (1 + modulation_index * message)
    am_signal = envelope * np.cos(2 * np.pi * carrier_freq * t)
    return t, am_signal, carrier, envelope

def am_power(carrier_power, modulation_index):
    """
    Calculate the total AM power for a single-tone modulating signal.

    Parameters:
    - carrier_power (float): Unmodulated carrier power (Pc).
    - modulation_index (float): Modulation index (m).

    Returns:
    - float: Total AM power.
    """
    return carrier_power * (1 + (modulation_index ** 2) / 2)

def am_multi_tone_power(carrier_power, mod_indices):
    """
    Calculate the total AM power for a multi-tone modulating signal.

    Parameters:
    - carrier_power (float): Unmodulated carrier power (Pc).
    - mod_indices (list or ndarray): Modulation indices for each tone.

    Returns:
    - float: Total AM power.
    """
    sum_m2 = np.sum(np.array(mod_indices) ** 2)
    return carrier_power * (1 + sum_m2 / 2)

def am_efficiency(modulation_index):
    """
    Calculate the power efficiency of a single-tone AM signal.

    Parameters:
    - modulation_index (float): Modulation index (m).

    Returns:
    - float: AM efficiency as a fraction.
    """
    m2 = modulation_index ** 2
    return m2 / (2 + m2)

def am_demodulate_envelope(am_signal, fs, cutoff_freq):
    """
    Demodulate an AM signal using envelope detection with the Hilbert transform.

    Parameters:
    - am_signal (ndarray): The AM signal.
    - fs (float): Sampling frequency in Hz.
    - cutoff_freq (float): Cutoff frequency for low-pass filtering.

    Returns:
    - ndarray: The demodulated signal.
    """
    analytic_signal = signal.hilbert(am_signal)
    envelope = np.abs(analytic_signal)
    
    # Low-pass filter to remove DC and high-frequency noise if necessary,
    # Here we perform basic DC blocking (AC coupling)
    demodulated = envelope - np.mean(envelope)
    
    # Optional low pass filtering (Butterworth)
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
    filtered = signal.filtfilt(b, a, demodulated)
    
    return filtered

def check_overmodulation(modulation_index):
    """
    Check if the AM modulation index causes overmodulation.

    Parameters:
    - modulation_index (float): Modulation index (m).

    Returns:
    - bool: True if overmodulated (m > 1.0), False otherwise.
    """
    return modulation_index > 1.0

def fm_modulate(message, carrier_freq, fs, freq_deviation, carrier_amp=1.0):
    """
    Generate an FM (Frequency Modulation) signal.

    Parameters:
    - message (ndarray): Modulating message signal.
    - carrier_freq (float): Carrier frequency in Hz.
    - fs (float): Sampling frequency in Hz.
    - freq_deviation (float): Maximum frequency deviation (Δf) in Hz.
    - carrier_amp (float): Carrier amplitude (Ac).

    Returns:
    - t (ndarray): Time vector.
    - fm_signal (ndarray): Modulated FM signal.
    """
    t = np.arange(len(message)) / fs
    # Integrate the message
    integral_message = np.cumsum(message) / fs
    
    phase = 2 * np.pi * carrier_freq * t + 2 * np.pi * freq_deviation * integral_message
    fm_signal = carrier_amp * np.cos(phase)
    
    return t, fm_signal

def fm_bandwidth_carson(freq_deviation, max_message_freq):
    """
    Calculate FM signal bandwidth using Carson's rule.

    Parameters:
    - freq_deviation (float): Maximum frequency deviation (Δf) in Hz.
    - max_message_freq (float): Maximum frequency in the message signal (fm) in Hz.

    Returns:
    - float: Estimated FM bandwidth.
    """
    return 2 * (freq_deviation + max_message_freq)

def fm_modulation_index(freq_deviation, max_message_freq):
    """
    Calculate the FM modulation index (β).

    Parameters:
    - freq_deviation (float): Maximum frequency deviation (Δf) in Hz.
    - max_message_freq (float): Maximum frequency in the message signal (fm) in Hz.

    Returns:
    - float: Modulation index (β).
    """
    return freq_deviation / max_message_freq

def fm_demodulate_discriminator(fm_signal, fs):
    """
    Demodulate an FM signal using a frequency discriminator (differentiation + envelope detection).

    Parameters:
    - fm_signal (ndarray): The FM signal.
    - fs (float): Sampling frequency in Hz.

    Returns:
    - ndarray: The demodulated signal.
    """
    # Differentiate the FM signal
    diff_fm = np.diff(fm_signal)
    diff_fm = np.append(diff_fm, 0)  # Match length
    
    # Envelope detection using Hilbert transform
    analytic_signal = signal.hilbert(diff_fm)
    envelope = np.abs(analytic_signal)
    
    # AC coupling (remove DC offset)
    demodulated = envelope - np.mean(envelope)
    
    return demodulated

def vsb_filter(signal_in, fs, carrier_freq, bandwidth, rolloff=0.25):
    """
    Apply a VSB (Vestigial Sideband) filter to an AM signal.
    
    Parameters:
    - signal_in (ndarray): Input signal.
    - fs (float): Sampling frequency.
    - carrier_freq (float): Carrier frequency.
    - bandwidth (float): Desired bandwidth for the VSB signal.
    - rolloff (float): Rolloff factor for the vestige.

    Returns:
    - ndarray: Filtered VSB signal.
    """
    # Design a simple bandpass filter mimicking a VSB characteristic.
    # In practice, a complex filter design is needed.
    # Here we use an asymmetrical bandpass filter around the carrier.
    nyquist = 0.5 * fs
    low_cutoff = (carrier_freq - bandwidth * rolloff) / nyquist
    high_cutoff = (carrier_freq + bandwidth) / nyquist
    
    if low_cutoff <= 0 or high_cutoff >= 1:
        raise ValueError("Filter cutoff frequencies must be between 0 and Nyquist frequency.")
        
    b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='band')
    vsb_sig = signal.filtfilt(b, a, signal_in)
    
    return vsb_sig

def plot_am(t, message, carrier, am_signal, envelope=None, title='AM Modulation'):
    """
    Create a multi-panel plot of AM signals.

    Returns:
    - fig (Figure): Matplotlib figure object.
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    axs[0].plot(t, message, color='blue')
    axs[0].set_title('Message Signal')
    axs[0].set_ylabel('Amplitude')
    axs[0].grid(True)
    
    axs[1].plot(t, carrier, color='green')
    axs[1].set_title('Carrier Signal')
    axs[1].set_ylabel('Amplitude')
    axs[1].grid(True)
    
    axs[2].plot(t, am_signal, color='red', label='AM Signal')
    if envelope is not None:
        axs[2].plot(t, envelope, color='black', linestyle='--', label='Envelope')
        axs[2].plot(t, -envelope, color='black', linestyle='--')
    axs[2].set_title(title)
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Amplitude')
    axs[2].legend()
    axs[2].grid(True)
    
    plt.tight_layout()
    return fig

def plot_fm(t, message, fm_signal, title='FM Modulation'):
    """
    Create a multi-panel plot of FM signals.

    Returns:
    - fig (Figure): Matplotlib figure object.
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    axs[0].plot(t, message, color='blue')
    axs[0].set_title('Message Signal')
    axs[0].set_ylabel('Amplitude')
    axs[0].grid(True)
    
    axs[1].plot(t, fm_signal, color='red')
    axs[1].set_title(title)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Amplitude')
    axs[1].grid(True)
    
    plt.tight_layout()
    return fig

def plot_spectrum(signal_in, fs, title='Frequency Spectrum'):
    """
    Plot the single-sided amplitude spectrum of a signal.

    Returns:
    - fig (Figure): Matplotlib figure object.
    """
    N = len(signal_in)
    freqs = np.fft.fftfreq(N, 1/fs)
    spectrum = np.fft.fft(signal_in)
    
    # Only keep positive frequencies
    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    amplitudes = np.abs(spectrum[pos_mask]) * 2 / N
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs, amplitudes, color='purple')
    ax.set_title(title)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.grid(True)
    
    plt.tight_layout()
    return fig
