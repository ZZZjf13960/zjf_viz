from .theme import set_theme, palette
from .plots import scatter, line, bar, box, heatmap, hist
from .timeseries import stacked_signals, time_frequency, psd
from .eeg import topoplot

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
    "topoplot"
]
