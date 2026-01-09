import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import griddata
from .utils import finalize_plot

# Approximate 2D positions for standard 10-20 system (normalized to unit circle)
# Nose is at (0, 1), Left Ear (-1, 0), Right Ear (1, 0)
# These are top-down view projections
STANDARD_1020 = {
    'Fp1': (-0.3, 0.8), 'Fp2': (0.3, 0.8), 'Fpz': (0, 0.8),
    'AF7': (-0.5, 0.75), 'AF3': (-0.25, 0.75), 'AFz': (0, 0.75), 'AF4': (0.25, 0.75), 'AF8': (0.5, 0.75),
    'F7': (-0.8, 0.5), 'F5': (-0.6, 0.5), 'F3': (-0.4, 0.5), 'F1': (-0.2, 0.5), 'Fz': (0, 0.5), 'F2': (0.2, 0.5), 'F4': (0.4, 0.5), 'F6': (0.6, 0.5), 'F8': (0.8, 0.5),
    'FT7': (-0.85, 0.25), 'FC5': (-0.65, 0.25), 'FC3': (-0.45, 0.25), 'FC1': (-0.15, 0.25), 'FCz': (0, 0.25), 'FC2': (0.15, 0.25), 'FC4': (0.45, 0.25), 'FC6': (0.65, 0.25), 'FT8': (0.85, 0.25),
    'T7': (-0.9, 0), 'C5': (-0.7, 0), 'C3': (-0.45, 0), 'C1': (-0.15, 0), 'Cz': (0, 0), 'C2': (0.15, 0), 'C4': (0.45, 0), 'C6': (0.7, 0), 'T8': (0.9, 0),
    'TP7': (-0.85, -0.25), 'CP5': (-0.65, -0.25), 'CP3': (-0.45, -0.25), 'CP1': (-0.15, -0.25), 'CPz': (0, -0.25), 'CP2': (0.15, -0.25), 'CP4': (0.45, -0.25), 'CP6': (0.65, -0.25), 'TP8': (0.85, -0.25),
    'P7': (-0.8, -0.5), 'P5': (-0.6, -0.5), 'P3': (-0.4, -0.5), 'P1': (-0.2, -0.5), 'Pz': (0, -0.5), 'P2': (0.2, -0.5), 'P4': (0.4, -0.5), 'P6': (0.6, -0.5), 'P8': (0.8, -0.5),
    'PO7': (-0.5, -0.75), 'PO3': (-0.25, -0.75), 'POz': (0, -0.75), 'PO4': (0.25, -0.75), 'PO8': (0.5, -0.75),
    'O1': (-0.3, -0.85), 'Oz': (0, -0.85), 'O2': (0.3, -0.85),
    'Iz': (0, -0.95)
}

# Aliases
STANDARD_1020.update({
    'T3': STANDARD_1020['T7'], 'T4': STANDARD_1020['T8'],
    'T5': STANDARD_1020['P7'], 'T6': STANDARD_1020['P8'] # Approx
})

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

def topoplot(data, montage=None, sensors=True, names=None, title=None, save_path=None, **kwargs):
    """
    Plots a topographic map of EEG data (without MNE dependency if montage provided manually).

    Args:
        data: 1D array-like of values or dict {channel_name: value}.
        montage: Dictionary of {channel_name: (x, y)} or list of (x, y) coordinates.
                 If None, tries to use names to look up in STANDARD_1020.
        sensors: Whether to plot sensor positions.
        names: List of channel names corresponding to data (if data is array).
        title: Plot title.
        save_path: Path to save the plot.
        **kwargs: Additional arguments for plt.imshow / griddata.
    """
    plt.figure()
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.axis('off')

    # Prepare data points
    points = []
    values = []

    if isinstance(data, dict):
        for name, val in data.items():
            if montage and name in montage:
                points.append(montage[name])
                values.append(val)
            elif name in STANDARD_1020:
                points.append(STANDARD_1020[name])
                values.append(val)
            else:
                print(f"Warning: Unknown channel position for {name}")
    else:
        # data is list/array
        if names is None:
            raise ValueError("Must provide 'names' if data is an array")

        for i, val in enumerate(data):
            name = names[i]
            if montage and name in montage:
                points.append(montage[name])
                values.append(val)
            elif name in STANDARD_1020:
                points.append(STANDARD_1020[name])
                values.append(val)
            else:
                # If montage is provided as list of coordinates
                if isinstance(montage, (list, np.ndarray)) and len(montage) == len(data):
                   points.append(montage[i])
                   values.append(val)
                else:
                   print(f"Warning: Unknown channel position for {name}")

    if not points:
        raise ValueError("No valid channel positions found to plot.")

    points = np.array(points)
    values = np.array(values)

    # Grid for interpolation
    grid_x, grid_y = np.mgrid[-1:1:100j, -1:1:100j]

    # Interpolate
    # 'cubic' looks smoother but can overshoot. 'linear' is safer but uglier.
    grid_z = griddata(points, values, (grid_x, grid_y), method='cubic', fill_value=np.nan)

    # Mask outside circle
    mask = (grid_x**2 + grid_y**2) > 1
    grid_z[mask] = np.nan

    # Plot heatmap
    im = ax.imshow(grid_z.T, extent=(-1, 1, -1, 1), origin='lower', cmap='coolwarm', **kwargs)

    # Add contours?
    # ax.contour(grid_x, grid_y, grid_z.T, colors='k', linewidths=0.5)

    # Draw head circle
    circle = patches.Circle((0, 0), 1, edgecolor='black', facecolor='none', linewidth=2)
    ax.add_patch(circle)

    # Draw nose
    nose = patches.Polygon([(-0.1, 0.98), (0, 1.1), (0.1, 0.98)], edgecolor='black', facecolor='white', linewidth=2)
    ax.add_patch(nose)

    # Draw ears
    # Simplify as ellipses or arcs
    # Left ear
    ax.add_patch(patches.Arc((-1, 0), 0.15, 0.3, theta1=90, theta2=270, color='black', linewidth=2))
    # Right ear
    ax.add_patch(patches.Arc((1, 0), 0.15, 0.3, theta1=-90, theta2=90, color='black', linewidth=2))

    if sensors:
        ax.scatter(points[:, 0], points[:, 1], c='black', s=10)

    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return finalize_plot(ax, title, None, None, save_path)

def plot_montage(montage=None, title="Sensor Layout", save_path=None):
    """
    Plots the sensor positions on a head.

    Args:
        montage: Dictionary or list of coordinates. If None, uses STANDARD_1020.
    """
    if montage is None:
        montage = STANDARD_1020

    return topoplot({k: 0 for k in montage}, montage=montage, sensors=True, title=title, save_path=save_path, alpha=0) # alpha=0 to hide heatmap

def plot_connectivity(con, names, montage=None, threshold=None, title=None, save_path=None):
    """
    Plots connectivity between sensors on a head.

    Args:
        con: (n_channels, n_channels) adjacency matrix.
        names: List of channel names.
        montage: Coordinate dictionary.
        threshold: Threshold to show connection.
    """
    # Reuse topoplot to draw head and sensors
    ax = topoplot({n: 0 for n in names}, montage=montage, sensors=True, names=names, title=title, alpha=0)

    # Get coordinates
    coords = []
    for name in names:
        if montage and name in montage:
            coords.append(montage[name])
        elif name in STANDARD_1020:
            coords.append(STANDARD_1020[name])
        else:
             # Fallback or error?
             coords.append((0,0))
    coords = np.array(coords)

    # Draw lines
    n_channels = len(names)
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            val = con[i, j]
            if threshold is None or abs(val) >= threshold:
                p1 = coords[i]
                p2 = coords[j]
                # Scale width or alpha by value?
                lw = 1 + abs(val) * 2
                alpha = min(1, abs(val))
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='blue', linewidth=lw, alpha=alpha)

    return finalize_plot(ax, title, None, None, save_path)
