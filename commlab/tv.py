import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    'raster_scan_params',
    'vsb_tv_bandwidth',
    'generate_raster_pattern',
    'plot_tv_spectrum',
    'plot_raster_scan'
]

def raster_scan_params(lines=525, fps=30, interlaced=True):
    """
    Calculate parameters for a TV raster scan system.
    
    Parameters:
    - lines: Total number of lines per frame (e.g., 525 for NTSC).
    - fps: Frames per second (e.g., 30 for NTSC).
    - interlaced: Boolean, whether the scan is interlaced.
    
    Returns:
    - Dictionary with horizontal frequency, vertical frequency, active lines, and aspect ratio.
    """
    h_freq = lines * fps
    v_freq = fps * 2 if interlaced else fps
    # NTSC active lines are typically ~483 for 525 total lines
    active_lines = int(lines * 0.92)
    return {
        'h_freq': h_freq,
        'v_freq': v_freq,
        'active_lines': active_lines,
        'aspect_ratio': '4:3'
    }

def vsb_tv_bandwidth(video_bw_mhz=4.2, vestigial_bw_mhz=0.75, audio_offset_mhz=4.5, guard_band_mhz=0.25):
    """
    Calculate total TV channel bandwidth parameters.
    
    Parameters:
    - video_bw_mhz: Full video bandwidth in MHz.
    - vestigial_bw_mhz: Vestigial lower sideband bandwidth in MHz.
    - audio_offset_mhz: Offset of audio carrier from picture carrier in MHz.
    - guard_band_mhz: Guard band in MHz.
    
    Returns:
    - Dictionary with bandwidth components.
    """
    total_bw = vestigial_bw_mhz + audio_offset_mhz + guard_band_mhz
    return {
        'video_bandwidth_mhz': video_bw_mhz,
        'vestigial_bandwidth_mhz': vestigial_bw_mhz,
        'audio_carrier_offset_mhz': audio_offset_mhz,
        'guard_band_mhz': guard_band_mhz,
        'total_bandwidth_mhz': total_bw
    }

def generate_raster_pattern(width=320, height=240, pattern='bars'):
    """
    Generate a 2D numpy array representing a TV test pattern.
    
    Parameters:
    - width: Image width in pixels.
    - height: Image height in pixels.
    - pattern: Type of pattern ('bars', 'checkerboard', 'gradient').
    
    Returns:
    - 2D numpy array of the pattern with values in [0, 1].
    """
    img = np.zeros((height, width))
    if pattern == 'bars':
        bar_width = width // 8
        for i in range(8):
            start = i * bar_width
            end = (i + 1) * bar_width if i < 7 else width
            img[:, start:end] = (7 - i) / 7.0
    elif pattern == 'checkerboard':
        block_size = min(width, height) // 8
        for y in range(height):
            for x in range(width):
                if ((x // block_size) + (y // block_size)) % 2 == 0:
                    img[y, x] = 1.0
    elif pattern == 'gradient':
        for x in range(width):
            img[:, x] = x / float(width - 1)
    else:
        raise ValueError("Pattern must be 'bars', 'checkerboard', or 'gradient'")
    return img

def plot_tv_spectrum(video_bw=4.2, vestigial=0.75, audio_offset=4.5):
    """
    Plot an idealized NTSC-style VSB TV channel spectrum.
    
    Parameters:
    - video_bw: Video bandwidth in MHz.
    - vestigial: Vestigial lower sideband width in MHz.
    - audio_offset: Offset of audio carrier from picture carrier in MHz.
    
    Returns:
    - matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Vestigial LSB
    ax.fill_between([-vestigial, 0], [0, 0], [1, 1], alpha=0.5, color='blue', label='Vestigial LSB')
    # Video USB
    ax.fill_between([0, video_bw], [0, 0], [1, 1], alpha=0.5, color='orange', label='Video USB')
    
    # Carriers
    ax.axvline(0, color='red', linestyle='-', linewidth=2, label='Picture Carrier')
    ax.axvline(audio_offset, color='green', linestyle='-', linewidth=2, label='Sound Carrier')
    
    ax.set_xlim(-vestigial - 0.5, audio_offset + 1)
    ax.set_ylim(0, 1.2)
    ax.set_xlabel('Frequency offset relative to Picture Carrier (MHz)')
    ax.set_ylabel('Relative Amplitude')
    ax.set_title('VSB TV Channel Spectrum (Idealized)')
    ax.legend()
    ax.grid(True)
    return fig

def plot_raster_scan(pattern_array, title='Raster Scan Pattern'):
    """
    Display a raster scan pattern.
    
    Parameters:
    - pattern_array: 2D numpy array representing the image.
    - title: Title of the plot.
    
    Returns:
    - matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pattern_array, cmap='gray', aspect='auto', vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis('off')
    fig.colorbar(im, ax=ax, label='Luminance')
    return fig
