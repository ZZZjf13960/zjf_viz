import mne
import numpy as np
import matplotlib.pyplot as plt
from .utils import finalize_plot

def create_info(ch_names, sfreq=100, ch_types='eeg', montage='standard_1020'):
    """
    Helper to create mne.Info object.

    Args:
        ch_names: List of channel names.
        sfreq: Sampling frequency.
        ch_types: Channel types.
        montage: Name of the montage to use (e.g., 'standard_1020').

    Returns:
        mne.Info object.
    """
    info = mne.create_info(ch_names, sfreq, ch_types)
    try:
        montage_obj = mne.channels.make_standard_montage(montage)
        info.set_montage(montage_obj)
    except ValueError as e:
        print(f"Warning: Montage '{montage}' not found or channel names do not match. Details: {e}")
    except Exception as e:
        print(f"Warning: Could not set montage. Details: {e}")
    return info

def plot_topomap(data, ch_names=None, info=None, title=None, save_path=None, **kwargs):
    """
    Wrapper for mne.viz.plot_topomap.

    Args:
        data: (n_channels,) array of values to plot.
        ch_names: List of channel names. Required if info is None.
        info: mne.Info object. If provided, ch_names is ignored.
        title: Plot title.
        save_path: Path to save the plot.
        **kwargs: Additional arguments for mne.viz.plot_topomap.
    """
    if info is None:
        if ch_names is None:
            raise ValueError("Must provide either info or ch_names.")
        info = create_info(ch_names)

    fig, ax = plt.subplots()
    mne.viz.plot_topomap(data, info, axes=ax, show=False, **kwargs)

    return finalize_plot(ax, title, None, None, save_path)

def plot_sensors(ch_names=None, info=None, title=None, save_path=None, **kwargs):
    """
    Wrapper for mne.viz.plot_sensors.

    Args:
        ch_names: List of channel names. Required if info is None.
        info: mne.Info object. If provided, ch_names is ignored.
        title: Plot title.
        save_path: Path to save the plot.
        **kwargs: Additional arguments for mne.viz.plot_sensors.
    """
    if info is None:
        if ch_names is None:
            raise ValueError("Must provide either info or ch_names.")
        info = create_info(ch_names)

    # Allow user to pass axes, otherwise create one for 'topomap' kind
    kind = kwargs.get('kind', 'topomap')

    if kind == 'topomap':
        if 'axes' not in kwargs:
             fig, ax = plt.subplots()
             kwargs['axes'] = ax
        else:
             ax = kwargs['axes']

    # We pass show=False
    mne.viz.plot_sensors(info, show=False, **kwargs)

    # Get the current axes if we didn't create it explicitly or if MNE created it (for 3d)
    ax = plt.gca()

    return finalize_plot(ax, title, None, None, save_path)
