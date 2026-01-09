import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .utils import finalize_plot

def stacked_signals(data, offset=None, labels=None, sampling_rate=None, start_time=0, title=None, xlabel="Time (s)", ylabel="Channels", save_path=None, **kwargs):
    """
    Plots multiple time series signals stacked vertically.
    Useful for EEG data.

    Args:
        data: pandas DataFrame where columns are channels and index is time,
              or numpy array (n_samples, n_channels).
        offset: Vertical offset between signals. If None, calculated automatically.
        labels: List of channel names if data is numpy array.
        sampling_rate: Sampling rate to generate time axis if data is numpy array.
        start_time: Start time for the time axis.
        title: Plot title.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        save_path: Path to save the plot.
        **kwargs: Additional arguments passed to plt.plot.
    """
    plt.figure()
    ax = plt.gca()

    if isinstance(data, pd.DataFrame):
        signals = data.values
        channel_names = data.columns.tolist()
        if sampling_rate:
            time = np.arange(signals.shape[0]) / sampling_rate + start_time
        else:
            # Try to use index as time if it's numeric
            try:
                time = data.index.to_numpy(dtype=float)
            except:
                time = np.arange(signals.shape[0])
    else:
        signals = data
        if labels:
            channel_names = labels
        else:
            channel_names = [f"Ch{i+1}" for i in range(signals.shape[1])]

        if sampling_rate:
            time = np.arange(signals.shape[0]) / sampling_rate + start_time
        else:
            time = np.arange(signals.shape[0])

    n_samples, n_channels = signals.shape

    # Calculate offset if not provided
    if offset is None:
        # A simple heuristic: 3 times the median standard deviation of channels
        stds = np.std(signals, axis=0)
        offset = np.median(stds) * 5
        if offset == 0:
            offset = 1.0 # Default if flat signals

    # Plot each channel with offset
    # We plot the first channel at the top (highest y) or bottom?
    # Usually Ch1 is at top. Let's make Ch1 at top.
    # So y for Ch_i = signal_i + (n_channels - 1 - i) * offset

    yticks = []
    for i in range(n_channels):
        # We want the first channel in the list to be at the top
        y_pos = (n_channels - 1 - i) * offset
        ax.plot(time, signals[:, i] + y_pos, **kwargs)
        yticks.append(y_pos)

    # Set y-ticks to channel names
    ax.set_yticks(yticks)
    ax.set_yticklabels(channel_names)

    # Adjust y limits to look nice
    # margins
    margin = offset * 0.5
    ax.set_ylim(-margin, (n_channels - 1) * offset + margin + np.max(np.abs(signals)))

    return finalize_plot(ax, title, xlabel, ylabel, save_path)
