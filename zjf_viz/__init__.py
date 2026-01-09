from .theme import set_theme, palette
from .plots import scatter, line, bar, box, heatmap, hist
from .timeseries import stacked_signals, time_frequency, psd, butterfly, erp_image
from .eeg import topoplot, plot_montage, plot_connectivity

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
    "topoplot",
    "plot_montage",
    "plot_connectivity"
]
