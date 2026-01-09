from .theme import set_theme, palette
from .plots import scatter, line, bar, box, heatmap, hist
from .timeseries import stacked_signals, time_frequency, psd, butterfly, erp_image
from .eeg import plot_topomap, plot_sensors, create_info, topoplot, plot_montage, plot_connectivity
from .neuro import plot_glass_brain, plot_stat_map, plot_connectome

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
    "time_frequency",
    "psd",
    "butterfly",
    "erp_image",
    "plot_topomap",
    "plot_sensors",
    "create_info",
    "topoplot",
    "plot_montage",
    "plot_connectivity",
    "plot_glass_brain",
    "plot_stat_map",
    "plot_connectome"
]
