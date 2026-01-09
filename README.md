# ZJF-viz

**ZJF-viz** is a Python library designed to make data visualization beautiful and convenient. It provides a high-level wrapper around `matplotlib` and `seaborn` with a custom, aesthetically pleasing theme by default.

## Features

- **Beautiful Defaults**: No need to tweak matplotlib rcParams manually.
- **Simple API**: Easy-to-use functions for common plot types.
- **Consistent Styling**: Uniform look and feel across all your visualizations.

## Installation

```bash
pip install .
```

## Usage

```python
import zjf_viz as zviz
import pandas as pd
import numpy as np

# Set the theme
zviz.set_theme()

# Create dummy data
df = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'category': np.random.choice(['A', 'B'], 100)
})

# Plot
zviz.scatter(data=df, x='x', y='y', hue='category', title="My Beautiful Scatter Plot")
```
