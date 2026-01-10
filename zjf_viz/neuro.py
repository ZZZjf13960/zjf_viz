from nilearn import plotting
import matplotlib.pyplot as plt

def plot_glass_brain(stat_map_img, title=None, save_path=None, **kwargs):
    """
    Wrapper for nilearn.plotting.plot_glass_brain.

    Args:
        stat_map_img: Nifti image or string path.
        title: Plot title.
        save_path: Path to save the plot.
        **kwargs: Additional arguments for nilearn.plotting.plot_glass_brain.
    """
    # nilearn creates its own figure usually, or accepts axes.
    # It returns a display object.
    display = plotting.plot_glass_brain(stat_map_img, title=title, **kwargs)

    if save_path:
        display.savefig(save_path)
    return display

def plot_stat_map(stat_map_img, bg_img=None, title=None, save_path=None, **kwargs):
    """
    Wrapper for nilearn.plotting.plot_stat_map.

    Args:
        stat_map_img: Nifti image or string path.
        bg_img: Background image.
        title: Plot title.
        save_path: Path to save the plot.
        **kwargs: Additional arguments for nilearn.plotting.plot_stat_map.
    """
    display = plotting.plot_stat_map(stat_map_img, bg_img=bg_img, title=title, **kwargs)

    if save_path:
        display.savefig(save_path)
    return display

def plot_connectome(adjacency_matrix, node_coords, node_color='auto', node_size=50, title=None, save_path=None, **kwargs):
    """
    Wrapper for nilearn.plotting.plot_connectome.

    Args:
        adjacency_matrix: Adjacency matrix of connections.
        node_coords: Coordinates of nodes.
        node_color: Color of nodes.
        node_size: Size of nodes.
        title: Plot title.
        save_path: Path to save the plot.
        **kwargs: Additional arguments for nilearn.plotting.plot_connectome.
    """
    display = plotting.plot_connectome(adjacency_matrix, node_coords, node_color=node_color, node_size=node_size, title=title, **kwargs)

    if save_path:
        display.savefig(save_path)
    return display
