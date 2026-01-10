import seaborn as sns
import matplotlib.pyplot as plt
from .utils import finalize_plot

def scatter(data, x, y, hue=None, size=None, style=None, title=None, xlabel=None, ylabel=None, save_path=None, **kwargs):
    """
    Draws a scatter plot.
    """
    plt.figure()
    ax = sns.scatterplot(data=data, x=x, y=y, hue=hue, size=size, style=style, **kwargs)
    return finalize_plot(ax, title, xlabel, ylabel, save_path)

def line(data, x, y, hue=None, style=None, title=None, xlabel=None, ylabel=None, save_path=None, **kwargs):
    """
    Draws a line plot.
    """
    plt.figure()
    ax = sns.lineplot(data=data, x=x, y=y, hue=hue, style=style, **kwargs)
    return finalize_plot(ax, title, xlabel, ylabel, save_path)

def bar(data, x, y, hue=None, title=None, xlabel=None, ylabel=None, save_path=None, **kwargs):
    """
    Draws a bar plot.
    """
    plt.figure()
    ax = sns.barplot(data=data, x=x, y=y, hue=hue, **kwargs)
    return finalize_plot(ax, title, xlabel, ylabel, save_path)

def box(data, x, y, hue=None, title=None, xlabel=None, ylabel=None, save_path=None, **kwargs):
    """
    Draws a box plot.
    """
    plt.figure()
    ax = sns.boxplot(data=data, x=x, y=y, hue=hue, **kwargs)
    return finalize_plot(ax, title, xlabel, ylabel, save_path)

def heatmap(data, title=None, xlabel=None, ylabel=None, save_path=None, **kwargs):
    """
    Draws a heatmap.
    """
    plt.figure()
    ax = sns.heatmap(data=data, annot=True, fmt=".2f", cmap="coolwarm", **kwargs)
    return finalize_plot(ax, title, xlabel, ylabel, save_path)

def hist(data, x, hue=None, kde=True, title=None, xlabel=None, ylabel=None, save_path=None, **kwargs):
    """
    Draws a histogram with optional KDE.
    """
    plt.figure()
    ax = sns.histplot(data=data, x=x, hue=hue, kde=kde, **kwargs)
    return finalize_plot(ax, title, xlabel, ylabel, save_path)
