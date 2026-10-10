"""
Delta Modulation (DM), Adaptive Delta Modulation (ADM), and Differential PCM (DPCM) module.
Implements encoding, decoding, and analysis for differential pulse code modulation techniques.
"""

import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    'dpcm_encode',
    'dpcm_decode',
    'dm_encode',
    'dm_decode',
    'adaptive_dm_encode',
    'adaptive_dm_decode',
    'detect_slope_overload',
    'detect_granular_noise',
    'plot_dm_analysis',
    'plot_adm_comparison'
]

def dpcm_encode(signal, predictor_order=1):
    """
    DPCM encoder with an nth-order linear predictor.
    
    Args:
        signal (ndarray): The input analog signal array.
        predictor_order (int): The order of the linear predictor.
        
    Returns:
        tuple: (prediction_error, predicted, quantized_error, codes)
    """
    N = len(signal)
    prediction_error = np.zeros(N)
    quantized_error = np.zeros(N)
    predicted = np.zeros(N)
    codes = np.zeros(N, dtype=int)
    
    # Use simple averaging filter for the predictor coefficients
    coeffs = np.ones(predictor_order) / predictor_order
    reconstructed = np.zeros(N)
    
    for i in range(N):
        if i == 0:
            pred = 0.0
        else:
            start = max(0, i - predictor_order)
            past_vals = reconstructed[start:i]
            # Dot product for linear prediction
            pred = np.dot(past_vals[::-1], coeffs[:len(past_vals)])
            
        predicted[i] = pred
        prediction_error[i] = signal[i] - pred
        
        # Simple integer round quantization for demonstration
        quantized_error[i] = np.round(prediction_error[i])
        codes[i] = int(quantized_error[i])
        
        # Reconstruct the sample as it would be at the decoder to prevent drift
        reconstructed[i] = pred + quantized_error[i]
        
    return prediction_error, predicted, quantized_error, codes

def dpcm_decode(codes, predictor_order=1):
    """
    DPCM decoder.
    
    Args:
        codes (ndarray): The encoded DPCM codes.
        predictor_order (int): The order of the linear predictor used in encoding.
        
    Returns:
        ndarray: The reconstructed signal.
    """
    N = len(codes)
    reconstructed = np.zeros(N)
    coeffs = np.ones(predictor_order) / predictor_order
    
    for i in range(N):
        if i == 0:
            pred = 0.0
        else:
            start = max(0, i - predictor_order)
            past_vals = reconstructed[start:i]
            pred = np.dot(past_vals[::-1], coeffs[:len(past_vals)])
            
        reconstructed[i] = pred + codes[i]
        
    return reconstructed

def dm_encode(signal, step_size):
    """
    Delta modulation encoder.
    
    Args:
        signal (ndarray): The input analog signal.
        step_size (float): The fixed step size for delta modulation.
        
    Returns:
        tuple: (dm_output, staircase_approx, error)
    """
    N = len(signal)
    dm_output = np.zeros(N)
    staircase_approx = np.zeros(N)
    error = np.zeros(N)
    
    current_approx = 0.0
    for i in range(N):
        error[i] = signal[i] - current_approx
        if error[i] >= 0:
            dm_output[i] = 1
            current_approx += step_size
        else:
            dm_output[i] = -1
            current_approx -= step_size
        staircase_approx[i] = current_approx
        
    return dm_output, staircase_approx, error

def dm_decode(dm_output, step_size):
    """
    DM decoder (accumulate steps).
    
    Args:
        dm_output (ndarray): The delta modulation output (-1 or 1).
        step_size (float): The fixed step size used for encoding.
        
    Returns:
        ndarray: The reconstructed staircase approximation signal.
    """
    return np.cumsum(dm_output * step_size)

def adaptive_dm_encode(signal, step_size_init=0.1, step_mult=2.0, step_div=2.0):
    """
    Adaptive DM (ADM) encoder: increases step size on consecutive same-sign errors,
    decreases on alternating errors.
    
    Args:
        signal (ndarray): The input analog signal.
        step_size_init (float): Initial step size.
        step_mult (float): Multiplier for step size increase.
        step_div (float): Divisor for step size decrease.
        
    Returns:
        tuple: (adm_output, staircase, step_sizes)
    """
    N = len(signal)
    adm_output = np.zeros(N)
    staircase = np.zeros(N)
    step_sizes = np.zeros(N)
    
    current_approx = 0.0
    current_step = step_size_init
    
    for i in range(N):
        error = signal[i] - current_approx
        
        if error >= 0:
            adm_output[i] = 1
        else:
            adm_output[i] = -1
            
        # Adapt step size based on previous output
        if i > 0:
            if adm_output[i] == adm_output[i-1]:
                current_step *= step_mult
            else:
                current_step /= step_div
                
        step_sizes[i] = current_step
        
        # Update staircase
        if adm_output[i] > 0:
            current_approx += current_step
        else:
            current_approx -= current_step
            
        staircase[i] = current_approx
        
    return adm_output, staircase, step_sizes

def adaptive_dm_decode(adm_output, step_size_init=0.1, step_mult=2.0, step_div=2.0):
    """
    Adaptive DM (ADM) decoder.
    
    Args:
        adm_output (ndarray): The encoded ADM sequence (-1 or 1).
        step_size_init (float): Initial step size.
        step_mult (float): Multiplier for step size increase.
        step_div (float): Divisor for step size decrease.
        
    Returns:
        ndarray: The reconstructed staircase signal.
    """
    N = len(adm_output)
    staircase = np.zeros(N)
    
    current_approx = 0.0
    current_step = step_size_init
    
    for i in range(N):
        if i > 0:
            if adm_output[i] == adm_output[i-1]:
                current_step *= step_mult
            else:
                current_step /= step_div
                
        if adm_output[i] > 0:
            current_approx += current_step
        else:
            current_approx -= current_step
            
        staircase[i] = current_approx
        
    return staircase

def detect_slope_overload(signal, staircase):
    """
    Detect regions of slope overload where the staircase approximation cannot track 
    the rapid changes in the signal.
    
    Args:
        signal (ndarray): Original signal.
        staircase (ndarray): Staircase approximation signal.
        
    Returns:
        ndarray: Boolean array where True indicates slope overload.
    """
    # Calculate difference step of staircase
    diff_staircase = np.abs(np.diff(staircase, prepend=staircase[0]))
    max_step = np.max(diff_staircase)
    if max_step == 0:
        max_step = 1.0 # fallback
        
    # Slope overload is characterized by the signal's rate of change exceeding
    # the maximum possible step size of the quantizer
    sig_diff = np.abs(np.diff(signal, prepend=signal[0]))
    return sig_diff > (max_step * 1.01)

def detect_granular_noise(signal, staircase, threshold=None):
    """
    Detect regions of granular noise where the staircase oscillates around a relatively flat signal.
    
    Args:
        signal (ndarray): Original signal.
        staircase (ndarray): Staircase approximation signal.
        threshold (float, optional): Maximum signal variation to be considered 'flat'.
        
    Returns:
        ndarray: Boolean array where True indicates granular noise.
    """
    diff_staircase = np.abs(np.diff(staircase, prepend=staircase[0]))
    max_step = np.max(diff_staircase)
    
    if threshold is None:
        threshold = max_step * 0.5
        
    sig_diff = np.abs(np.diff(signal, prepend=signal[0]))
    # Granular noise occurs when the signal is almost constant but the staircase oscillates
    return sig_diff < threshold

def plot_dm_analysis(t, original, staircase, slope_overload=None, granular=None, title='DM Analysis'):
    """
    Plot original vs staircase with slope overload and granular noise regions highlighted.
    
    Args:
        t (ndarray): Time array.
        original (ndarray): Original analog signal.
        staircase (ndarray): Staircase approximation signal.
        slope_overload (ndarray, optional): Boolean array of slope overload regions.
        granular (ndarray, optional): Boolean array of granular noise regions.
        title (str): Title of the plot.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, original, label='Original', color='blue')
    ax.step(t, staircase, where='post', label='Staircase Approx', color='orange')
    
    if slope_overload is not None:
        ax.fill_between(t, np.min(original), np.max(original), where=slope_overload, 
                        color='red', alpha=0.2, label='Slope Overload')
                        
    if granular is not None:
        ax.fill_between(t, np.min(original), np.max(original), where=granular, 
                        color='gray', alpha=0.3, label='Granular Noise')
                        
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.legend(loc='upper right')
    ax.grid(True)
    
    plt.tight_layout()
    return fig

def plot_adm_comparison(t, original, dm_staircase, adm_staircase, title='DM vs ADM Comparison'):
    """
    Side-by-side comparison of standard DM vs Adaptive DM.
    
    Args:
        t (ndarray): Time array.
        original (ndarray): Original analog signal.
        dm_staircase (ndarray): Standard delta modulation staircase.
        adm_staircase (ndarray): Adaptive delta modulation staircase.
        title (str): Title for the figure.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # DM Plot
    axes[0].plot(t, original, label='Original', color='blue')
    axes[0].step(t, dm_staircase, where='post', label='DM Staircase', color='orange')
    axes[0].set_title('Standard Delta Modulation (DM)')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend(loc='upper right')
    axes[0].grid(True)
    
    # ADM Plot
    axes[1].plot(t, original, label='Original', color='blue')
    axes[1].step(t, adm_staircase, where='post', label='ADM Staircase', color='green')
    axes[1].set_title('Adaptive Delta Modulation (ADM)')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Amplitude')
    axes[1].legend(loc='upper right')
    axes[1].grid(True)
    
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    return fig
