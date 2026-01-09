import zjf_viz as zviz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os

def run_demo():
    print("Running ZJF-viz demo...")
    os.makedirs("examples", exist_ok=True)

    # 1. Set the beautiful theme
    zviz.set_theme()

    # 2. Generate sample data
    np.random.seed(123)
    n = 100
    df = pd.DataFrame({
        'feature_1': np.random.randn(n),
        'feature_2': np.random.randn(n) * 2 + 1,
        'group': np.random.choice(['Control', 'Treatment'], n),
        'time': np.linspace(0, 10, n),
        'value': np.sin(np.linspace(0, 10, n)) + np.random.normal(0, 0.1, n)
    })

    # 3. Create Basic Plots

    # Scatter Plot
    print("Generating Scatter Plot...")
    zviz.scatter(df, x='feature_1', y='feature_2', hue='group',
                 title="Feature Correlation", xlabel="Feature 1", ylabel="Feature 2",
                 save_path="examples/scatter_demo.png")

    # Line Plot
    print("Generating Line Plot...")
    zviz.line(df, x='time', y='value',
              title="Time Series Analysis", xlabel="Time (s)", ylabel="Signal",
              save_path="examples/line_demo.png")

    # Bar Plot
    print("Generating Bar Plot...")
    # Aggregating for bar plot to make sense
    df_agg = df.groupby('group')['feature_2'].mean().reset_index()
    zviz.bar(df_agg, x='group', y='feature_2',
             title="Average Feature 2 by Group", xlabel="Group", ylabel="Mean Value",
             save_path="examples/bar_demo.png")

    # Histogram
    print("Generating Histogram...")
    zviz.hist(df, x='feature_1', hue='group', kde=True,
              title="Distribution of Feature 1",
              save_path="examples/hist_demo.png")

    # 4. Create EEG/Time-Series Plots
    print("Generating EEG Plots...")

    # Mock EEG Data (numpy array)
    n_channels = 32
    n_samples = 1000
    sampling_rate = 250
    eeg_data = np.random.randn(n_samples, n_channels)
    # Add some sine waves
    t = np.linspace(0, 4, n_samples)
    for i in range(n_channels):
        eeg_data[:, i] += np.sin(2 * np.pi * 10 * t + i/10)

    # Stacked Signals
    zviz.stacked_signals(eeg_data, sampling_rate=sampling_rate, title="EEG Stacked Signals",
                         save_path="examples/eeg_stacked.png")

    # Butterfly Plot
    zviz.butterfly(eeg_data, sampling_rate=sampling_rate, title="Butterfly Plot",
                   save_path="examples/eeg_butterfly.png")

    # Topoplot (using random data for standard channels)
    # Get standard names
    # Note: users might not know internal variables, but let's assume they have names
    # We can peek at zjf_viz.eeg.STANDARD_1020 keys
    from zjf_viz.eeg import STANDARD_1020
    available_channels = list(STANDARD_1020.keys())
    # Pick a subset
    plot_channels = available_channels[:20]
    topo_data = np.random.randn(len(plot_channels))
    zviz.topoplot(topo_data, names=plot_channels, title="EEG Topography",
                  save_path="examples/eeg_topo.png")

    # Spectrogram
    # Create a chirp
    chirp = np.sin(2 * np.pi * 10 * t + 2 * np.pi * 50 * t**2 / 4)
    zviz.time_frequency(chirp, sampling_rate, title="Spectrogram (Chirp)",
                        save_path="examples/spectrogram.png")

    # PSD
    zviz.psd(chirp, sampling_rate, title="Power Spectral Density",
             save_path="examples/psd.png")

    # 5. Neuroimaging Plots
    print("Generating Neuroimaging Plots...")

    # Create dummy Nifti
    affine = np.eye(4)
    affine[0, 0] = 2
    affine[1, 1] = 2
    affine[2, 2] = 2
    affine[:3, 3] = [-90, -126, -72] # Rough MNI origin shift

    vol_data = np.zeros((91, 109, 91))
    # Add a "blob" at the center
    cx, cy, cz = 45, 54, 45
    for x in range(91):
        for y in range(109):
            for z in range(91):
                if (x-cx)**2 + (y-cy)**2 + (z-cz)**2 < 100: # Radius 10
                    vol_data[x, y, z] = 1.0

    dummy_img = nib.Nifti1Image(vol_data, affine)

    # Glass Brain
    zviz.plot_glass_brain(dummy_img, title="Glass Brain (Dummy)",
                          save_path="examples/glass_brain.png")

    # Stat Map
    zviz.plot_stat_map(dummy_img, title="Stat Map (Dummy)",
                       save_path="examples/stat_map.png")

    print("Demo completed. Images saved in examples/ directory.")

if __name__ == "__main__":
    run_demo()
