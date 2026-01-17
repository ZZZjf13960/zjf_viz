from .theme import set_theme, palette
from .plots import scatter, line, bar, box, heatmap, hist
from .timeseries import stacked_signals
from .eeg import plot_topomap, plot_sensors, create_info
from .neuro import plot_glass_brain, plot_stat_map, plot_connectome

__version__ = "0.1.0"

__all__ = [
    "set_theme",
    "palette",
    "scatter",
    "line",
    "bar",
    "box",
    "heatmap",
    "hist",
    "stacked_signals",
    "plot_topomap",
    "plot_sensors",
    "create_info",
    "plot_glass_brain",
    "plot_stat_map",
    "plot_connectome"
]
