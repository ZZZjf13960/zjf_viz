import seaborn as sns
import matplotlib.pyplot as plt

def set_theme(style="whitegrid", font_scale=1.2, palette_name="deep"):
    """
    Sets the global theme for visualizations.

    Args:
        style (str): Seaborn style (e.g., "whitegrid", "darkgrid", "ticks").
        font_scale (float): Scaling factor for font size.
        palette_name (str): Name of the color palette.
    """
    sns.set_theme(style=style, font_scale=font_scale)

    # Custom tweaks for "beauty"
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['figure.dpi'] = 120
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.labelweight'] = 'medium'
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['legend.title_fontsize'] = 13

    # Remove top and right spines by default for a cleaner look
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

    # Set default palette
    sns.set_palette(palette_name)

def palette(n_colors=10, name="deep"):
    """Returns the current color palette."""
    return sns.color_palette(name, n_colors)
