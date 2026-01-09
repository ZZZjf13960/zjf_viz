import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
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

def time_frequency(data, sampling_rate, method='spectrogram', title=None, xlabel="Time (s)", ylabel="Frequency (Hz)", save_path=None, **kwargs):
    """
    Plots the time-frequency representation (spectrogram) of a signal.

    Args:
        data: 1D numpy array of signal data.
        sampling_rate: Sampling frequency in Hz.
        method: 'spectrogram' (STFT) or 'scalogram' (CWT - TODO). Currently supports 'spectrogram'.
        title: Plot title.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        save_path: Path to save the plot.
        **kwargs: Additional arguments for scipy.signal.spectrogram.
    """
    plt.figure()
    ax = plt.gca()

    if method == 'spectrogram':
        f, t, Sxx = signal.spectrogram(data, sampling_rate, **kwargs)
        # Use pcolormesh for better visualization
        # Use a logarithmic scale for power? Usually better.
        # But let's stick to linear or dB. Let's do 10*log10.
        Sxx_log = 10 * np.log10(Sxx + 1e-10) # Avoid log(0)

        im = ax.pcolormesh(t, f, Sxx_log, shading='gouraud', cmap='viridis')
        plt.colorbar(im, ax=ax, label='Power (dB/Hz)')
    else:
        raise NotImplementedError(f"Method {method} not implemented yet.")

    return finalize_plot(ax, title, xlabel, ylabel, save_path)

def psd(data, sampling_rate, method='welch', title=None, xlabel="Frequency (Hz)", ylabel="Power Spectral Density (V**2/Hz)", save_path=None, **kwargs):
    """
    Plots the Power Spectral Density (PSD) of a signal.

    Args:
        data: 1D numpy array of signal data.
        sampling_rate: Sampling frequency in Hz.
        method: 'welch' or 'periodogram'. Default is 'welch'.
        title: Plot title.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        save_path: Path to save the plot.
        **kwargs: Additional arguments for scipy.signal.welch.
    """
    plt.figure()
    ax = plt.gca()

    if method == 'welch':
        f, Pxx = signal.welch(data, sampling_rate, **kwargs)
        ax.semilogy(f, Pxx)
    elif method == 'periodogram':
        f, Pxx = signal.periodogram(data, sampling_rate, **kwargs)
        ax.semilogy(f, Pxx)
    else:
        raise NotImplementedError(f"Method {method} not implemented yet.")

    return finalize_plot(ax, title, xlabel, ylabel, save_path)

def butterfly(data, sampling_rate=None, start_time=0, title=None, xlabel="Time (s)", ylabel="Amplitude", color='black', alpha=0.5, save_path=None, **kwargs):
    """
    Plots all signals overlaid (butterfly plot).

    Args:
        data: (n_samples, n_channels) numpy array or DataFrame.
        sampling_rate: Sampling frequency.
        start_time: Start time.
        title: Plot title.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        color: Color of the lines.
        alpha: Transparency of the lines.
        save_path: Path to save.
    """
    plt.figure()
    ax = plt.gca()

    if isinstance(data, pd.DataFrame):
        signals = data.values
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
        if sampling_rate:
            time = np.arange(signals.shape[0]) / sampling_rate + start_time
        else:
            time = np.arange(signals.shape[0])

    ax.plot(time, signals, color=color, alpha=alpha, **kwargs)

    return finalize_plot(ax, title, xlabel, ylabel, save_path)

def erp_image(data, sampling_rate=None, start_time=0, title=None, xlabel="Time (s)", ylabel="Epochs", cmap="RdBu_r", save_path=None, **kwargs):
    """
    Plots an ERP image (heatmap of epochs x time) for a single channel.

    Args:
        data: (n_epochs, n_times) numpy array.
        sampling_rate: Sampling frequency.
        start_time: Start time.
        title: Title.
        xlabel: X label.
        ylabel: Y label.
        cmap: Colormap.
        save_path: Path to save.
    """
    plt.figure()
    ax = plt.gca()

    n_epochs, n_times = data.shape

    if sampling_rate:
        extent = [start_time, start_time + n_times / sampling_rate, 0, n_epochs]
    else:
        extent = [0, n_times, 0, n_epochs]

    im = ax.imshow(data, aspect='auto', origin='lower', cmap=cmap, extent=extent, **kwargs)
    plt.colorbar(im, ax=ax)

    return finalize_plot(ax, title, xlabel, ylabel, save_path)
