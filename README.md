# ZJF-viz

**ZJF-viz** is a comprehensive Python library designed to simplify data visualization. It provides a high-level interface for creating beautiful charts, EEG topographies, and neuroimaging plots with minimal boilerplate code.

## Features

- **General Visualization**: Streamlined wrappers for Scatter, Line, Bar, Box, Heatmap, and Histogram plots (powered by `seaborn` & `matplotlib`).
- **EEG Analysis**: Built-in support for plotting EEG topographies and sensor locations (powered by `mne`).
- **Neuroimaging**: Easy-to-use functions for glass brain, statistical maps, and connectome plotting (powered by `nilearn`).
- **Time Series**: Tools for stacking multiple signals for easy comparison.
- **Beautiful Defaults**: Publication-ready aesthetics out of the box.

## Installation

You can install `zjf_viz` and its dependencies using pip:

```bash
pip install .
```

*Note: Depending on your environment, you may need to install system dependencies for `mne` or `nilearn`.*

## Usage

### 1. General Plotting

Make standard plots beautiful with a single function call.

```python
import zjf_viz as zviz
import pandas as pd
import numpy as np

# Set the custom theme
zviz.set_theme()

# Create sample data
df = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'group': np.random.choice(['A', 'B'], 100)
})

# Scatter Plot
zviz.scatter(
    data=df, x='x', y='y', hue='group',
    title="My Scatter Plot",
    save_path="scatter.png"
)

# Line Plot
zviz.line(
    data=df, x='x', y='y',
    title="My Line Plot"
)
```

### 2. Time Series & Signals

Visualizing stacked time-series data (e.g., EEG channels) is straightforward.

```python
# signals: (n_samples, n_channels)
zviz.stacked_signals(
    signals=my_signal_array,
    sampling_rate=100,
    title="Stacked EEG Signals"
)
```

### 3. EEG Topography

Visualize scalp distributions easily.

```python
# Create info object (or use existing mne.Info)
info = zviz.create_info(ch_names=['Fz', 'Cz', 'Pz'], sfreq=100)

# Plot topomap
zviz.plot_topomap(
    data=my_data_vector,
    info=info,
    title="Alpha Power"
)
```

### 4. Neuroimaging

Visualize NIfTI images or connectivity matrices.

```python
# Glass Brain
zviz.plot_glass_brain(
    stat_map_img='my_stat_map.nii.gz',
    title="Activation Map"
)

# Connectome
zviz.plot_connectome(
    adjacency_matrix=my_adj_matrix,
    node_coords=my_coords,
    title="Functional Connectivity"
)
```

## Running Examples

The repository comes with a full example script generating various plots.

```bash
python3 examples/example.py
```

This will generate a set of demo images in the `examples/` directory.

## License

MIT
