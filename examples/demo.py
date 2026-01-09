import zjf_viz as zviz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_demo():
    print("Running ZJF-viz demo...")

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

    # 3. Create Plots

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

    print("Demo completed. Images saved in examples/ directory.")

if __name__ == "__main__":
    run_demo()
