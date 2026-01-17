# ZJF-viz

**ZJF-viz** is a Python library designed to make data visualization beautiful, convenient, and publication-ready. It bridges the gap between general-purpose plotting (via `seaborn` and `matplotlib`) and specialized neuroimaging/EEG visualization (via `mne` and `nilearn`), providing a unified and aesthetic interface.

## Features

-   **General Plotting**: High-level wrappers for common plots (scatter, line, bar, box, heatmap, hist) with a polished default theme.
-   **EEG Visualization**: Simplified plotting for EEG sensors and topomaps using `mne`.
-   **Neuroimaging**: Easy-to-use functions for brain glass brains, statistical maps, and connectomes using `nilearn`.
-   **Time-Series**: specialized plots like stacked signal traces.
-   **Convenience**: Integrated title, label, and file saving arguments in every function.

## Installation

You can install the package directly from the source:

```bash
pip install .
```

Dependencies include: `matplotlib`, `seaborn`, `pandas`, `numpy`, `mne`, `nilearn`, `nibabel`, `scikit-learn`.

## Usage

### 1. General Plotting

ZJF-viz wraps seaborn plots with a cleaner default aesthetic.

```python
import zjf_viz as zviz
import pandas as pd
import numpy as np

# Set the beautiful theme
zviz.set_theme()

# Create dummy data
df = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'category': np.random.choice(['Group A', 'Group B'], 100)
})

# Scatter Plot
zviz.scatter(
    data=df,
    x='x', y='y',
    hue='category',
    title="My Scatter Plot",
    save_path="scatter.png"
)

# Line Plot
zviz.line(
    data=df,
    x='x', y='y',
    hue='category',
    title="Trend Line",
    save_path="line.png"
)

# Bar Plot
zviz.bar(
    data=df,
    x='category', y='y',
    title="Category Comparison"
)
```

Other available plots: `zviz.box`, `zviz.heatmap`, `zviz.hist`.

### 2. EEG Visualization

Visualize EEG sensor locations and topomaps easily.

```python
import numpy as np
import zjf_viz as zviz

# Dummy data for 19 channels
data = np.random.rand(19)
ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'T3', 'C3', 'Cz', 'C4', 'T4',
            'T5', 'P3', 'Pz', 'P4', 'T6',
            'O1', 'O2']

# Plot Topomap
zviz.plot_topomap(
    data,
    ch_names=ch_names,
    title="Alpha Power Topomap",
    save_path="topomap.png"
)

# Plot Sensor Locations
zviz.plot_sensors(
    ch_names=ch_names,
    kind='3d',
    title="Sensor Layout"
)
```

### 3. Neuroimaging

Visualize NIfTI images on brain templates.

```python
import zjf_viz as zviz

# Path to your NIfTI file
stat_img = 'path/to/stat_map.nii.gz'

# Glass Brain Projection
zviz.plot_glass_brain(
    stat_img,
    title="Activation Map",
    display_mode='lzry',
    colorbar=True
)

# Statistical Map on Standard Brain
zviz.plot_stat_map(
    stat_img,
    title="Stat Map",
    threshold=3.0
)
```

### 4. Time-Series Stacking

Visualize multiple signals stacked vertically (common in EEG/MEG raw data inspection).

```python
import numpy as np
import zjf_viz as zviz

# Generate 3 signals
signals = np.random.randn(1000, 3) # 1000 samples, 3 channels

zviz.stacked_signals(
    signals,
    sampling_rate=250,
    channel_names=['Ch1', 'Ch2', 'Ch3'],
    title="Raw Signals",
    figsize=(12, 6)
)
```

## API Reference

### Theme
-   `set_theme(style="whitegrid", font_scale=1.2, palette_name="deep")`: Sets the global plotting theme.

### Plots
All plot functions accept `title`, `xlabel`, `ylabel`, and `save_path`.
-   `scatter(data, x, y, hue=...)`
-   `line(data, x, y, hue=...)`
-   `bar(data, x, y, hue=...)`
-   `box(data, x, y, hue=...)`
-   `heatmap(data, ...)`
-   `hist(data, x, hue=...)`

### EEG (`zjf_viz.eeg`)
-   `plot_topomap(data, ch_names=None, info=None, ...)`
-   `plot_sensors(ch_names=None, info=None, ...)`

### Neuro (`zjf_viz.neuro`)
-   `plot_glass_brain(stat_map_img, ...)`
-   `plot_stat_map(stat_map_img, ...)`
-   `plot_connectome(adjacency_matrix, node_coords, ...)`

### Time-Series (`zjf_viz.timeseries`)
-   `stacked_signals(signals, sampling_rate, ...)`
