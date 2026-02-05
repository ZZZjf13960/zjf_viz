import zjf_viz as zviz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os

def run_demo():
    print("Running ZJF-viz full demo...")

    # Ensure examples directory exists
    os.makedirs("examples", exist_ok=True)

    # 1. Set the beautiful theme
    zviz.set_theme()

    # --- Basic Plots ---
    print("\n--- Generating Basic Plots ---")

    # Generate sample data
    np.random.seed(123)
    n = 100
    df = pd.DataFrame({
        'feature_1': np.random.randn(n),
        'feature_2': np.random.randn(n) * 2 + 1,
        'group': np.random.choice(['Control', 'Treatment'], n),
        'time': np.linspace(0, 10, n),
        'value': np.sin(np.linspace(0, 10, n)) + np.random.normal(0, 0.1, n)
    })

    # Scatter Plot
    print("Scatter Plot...")
    zviz.scatter(df, x='feature_1', y='feature_2', hue='group',
                 title="Feature Correlation", xlabel="Feature 1", ylabel="Feature 2",
                 save_path="examples/scatter_demo.png")

    # Line Plot
    print("Line Plot...")
    zviz.line(df, x='time', y='value',
              title="Time Series Analysis", xlabel="Time (s)", ylabel="Signal",
              save_path="examples/line_demo.png")

    # Bar Plot
    print("Bar Plot...")
    # Aggregating for bar plot to make sense
    df_agg = df.groupby('group')['feature_2'].mean().reset_index()
    zviz.bar(df_agg, x='group', y='feature_2',
             title="Average Feature 2 by Group", xlabel="Group", ylabel="Mean Value",
             save_path="examples/bar_demo.png")

    # Box Plot
    print("Box Plot...")
    zviz.box(df, x='group', y='feature_1', hue='group',
             title="Feature 1 Distribution by Group",
             save_path="examples/box_demo.png")

    # Heatmap
    print("Heatmap...")
    corr = df[['feature_1', 'feature_2', 'value']].corr()
    zviz.heatmap(corr, title="Correlation Matrix",
                 save_path="examples/heatmap_demo.png")

    # Histogram
    print("Histogram...")
    zviz.hist(df, x='feature_1', hue='group', kde=True,
              title="Distribution of Feature 1",
              save_path="examples/hist_demo.png")


    # --- Time Series / EEG Signal Plots ---
    print("\n--- Generating Time Series / EEG Signal Plots ---")

    # Generate dummy EEG-like data
    n_channels = 5
    n_samples = 200
    sfreq = 100
    t = np.arange(n_samples) / sfreq
    signals = np.zeros((n_samples, n_channels))
    for i in range(n_channels):
        signals[:, i] = np.sin(2 * np.pi * (i + 1) * t) + np.random.normal(0, 0.2, n_samples)

    # Stacked Signals
    print("Stacked Signals...")
    zviz.stacked_signals(signals, sampling_rate=sfreq, title="EEG Stacked Signals",
                         save_path="examples/stacked_signals_demo.png")


    # --- EEG Topography ---
    print("\n--- Generating EEG Topography Plots ---")

    # Create info
    ch_names = ['Fz', 'Cz', 'Pz', 'C3', 'C4']
    info = zviz.create_info(ch_names, sfreq=sfreq, montage='standard_1020')

    # Dummy data for topography (one value per channel)
    topo_data = np.random.randn(len(ch_names))

    # Plot Topomap
    print("Topomap...")
    zviz.plot_topomap(topo_data, info=info, title="Alpha Power Topography",
                      save_path="examples/topomap_demo.png")

    # Plot Sensors
    print("Sensor Locations...")
    zviz.plot_sensors(info=info, title="Sensor Locations",
                      save_path="examples/sensors_demo.png")


    # --- Neuroimaging Plots ---
    print("\n--- Generating Neuroimaging Plots ---")

    # Create a dummy Nifti image (3D random noise)
    # Using small dimensions to keep it fast
    data_shape = (91, 109, 91) # Standard MNI shape roughly, but let's make it smaller for speed?
    # Actually nilearn plotting usually expects MNI space.
    # To avoid complex registration issues in a demo without real data,
    # we can try to use nilearn's MNI template if available, or just random noise with an identity affine.
    # However, plot_glass_brain expects data in MNI space usually.
    # Let's create a random image with an identity affine and hope nilearn handles it gracefully or warns.
    # Better: create a "blob" in the center to make it look like something.

    affine = np.eye(4)
    # Scale it up to match roughly mm space if we assume identity is 1mm
    # MNI origin is roughly at the center.
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
    print("Glass Brain...")
    # Note: glass_brain might warn or fail if the affine is weird, but let's try.
    zviz.plot_glass_brain(dummy_img, title="Glass Brain Activation (Dummy)",
                          save_path="examples/glass_brain_demo.png")

    # Stat Map
    print("Stat Map...")
    zviz.plot_stat_map(dummy_img, title="Stat Map (Dummy)",
                       save_path="examples/stat_map_demo.png")

    # Connectome
    print("Connectome...")
    # 4 random nodes
    coords = np.array([
        [0, 0, 0],
        [20, 20, 20],
        [-20, -20, 0],
        [20, -20, 10]
    ])
    adj_matrix = np.array([
        [0, 1, 0, 0.5],
        [1, 0, 0.2, 0],
        [0, 0.2, 0, 0.8],
        [0.5, 0, 0.8, 0]
    ])
    zviz.plot_connectome(adj_matrix, coords, title="Functional Connectivity",
                         save_path="examples/connectome_demo.png")

    print("\nDemo completed. All examples generated in examples/ directory.")

if __name__ == "__main__":
    run_demo()
