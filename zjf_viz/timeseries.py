import numpy as np
import matplotlib.pyplot as plt
from .utils import finalize_plot

def stacked_signals(signals, sampling_rate=1.0, channel_names=None, title=None, save_path=None, **kwargs):
    """
    Plots multiple time series signals stacked vertically.

    Args:
        signals: (n_samples, n_channels) numpy array of signal data.
        sampling_rate: Sampling frequency in Hz.
        channel_names: List of channel names. If None, channels are numbered.
        title: Plot title.
        save_path: Path to save the plot.
        **kwargs: Additional arguments.
    """
    n_samples, n_channels = signals.shape
    time = np.arange(n_samples) / sampling_rate

    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 8)))

    # Calculate offset to stack signals
    # A simple heuristic: offset by max range of any signal
    # Or typically, we might want them equally spaced.
    # Let's normalize signals or just space them out.
    # Here we space them by the standard deviation * some factor, or max-min.

    # Let's find a global scale or per-channel scale
    # To mimic standard EEG plot, we usually offset by a fixed amount.

    # Calculate offsets
    # We want channel 0 at the top or bottom? Usually top is channel 0 in EEG.
    # But y-axis usually increases upwards. So channel 0 at top means highest y.

    # Let's check the range of the data
    data_min = np.min(signals)
    data_max = np.max(signals)
    data_range = data_max - data_min

    # If data_range is 0 (flat signals), use 1.0
    if data_range == 0:
        data_range = 1.0

    # Better heuristic: Compute robust range for each channel or global
    # For simplicity, let's just stack them with a fixed gap.
    # Gap = mean peak-to-peak amplitude
    ptp = np.ptp(signals, axis=0)
    mean_ptp = np.mean(ptp)
    if mean_ptp == 0:
        mean_ptp = 1.0

    spacing = mean_ptp * 1.5 # Add some buffer

    offsets = np.arange(n_channels) * spacing
    # Reverse so first channel is at top
    offsets = offsets[::-1]

    for i in range(n_channels):
        ax.plot(time, signals[:, i] + offsets[i], color='k', linewidth=0.8)

    ax.set_yticks(offsets)
    if channel_names:
        ax.set_yticklabels(channel_names)
    else:
        ax.set_yticklabels([f"Ch {i}" for i in range(n_channels)])

    ax.set_xlabel("Time (s)")

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return finalize_plot(ax, title, None, None, save_path)
