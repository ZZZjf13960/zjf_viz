import matplotlib.pyplot as plt

def finalize_plot(ax, title=None, xlabel=None, ylabel=None, save_path=None):
    """
    Common helper to finalize plots with titles, labels, and saving.
    """
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    return ax
