from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
import numpy as np


def plot_samples(
        draws: np.ndarray,
        image_shape: Optional[Tuple[int, int]] = None,
        columns: int = 4,
        rows: int = 3,
        figsize: Tuple[int, int] = (19, 12),
        title: str = 'CAR priors',
        custom_min: int = None,
        custom_max: int = None) -> None:
    if image_shape is None:
        image_shape = (15, 10)  # (num_x, num_y)

    fig, axes = plt.subplots(rows, columns, figsize=figsize)

    num_x, num_y = image_shape
    for r in range(rows):
        for c in range(columns):
            im = axes[r, c].imshow(
                draws[r*columns + c].reshape(num_y, num_x),
                # extent=[0,1,0,1],
                cmap='viridis',
                interpolation='none',
                origin='lower',
                vmin=custom_min,
                vmax=custom_max)
            axes[r, c].set_title(f'draw {str(r*columns + c)}')
            fig.colorbar(im, ax=axes[r, c])
    fig.suptitle(title, fontsize=16)
    plt.show()


def plot_images_from_arrays(
        arrays: List[np.ndarray],
        titles: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 4),
        image_shape: Optional[Tuple[int, int]] = None,
        cmap: str = 'viridis',
        flip: bool = False,
        custom_min: int = None,
        custom_max: int = None) -> None:
    if image_shape is None:
        image_shape = (15, 10)  # (num_x, num_y)

    nrows = 1
    ncols = len(arrays)
    
    if flip:
        ncols, nrows = nrows, ncols
        
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if custom_min is None:
        custom_min = np.amin(arrays)   
    if custom_max is None:
        custom_max = np.amax(arrays)   

    imshow_dict = {
        'cmap': cmap,
        'interpolation': 'none',
        'origin': 'lower',
        'vmin': custom_min,
        'vmax': custom_max,
    }
    num_x, num_y = image_shape
    if len(arrays) == 1:
        axes = [axes]
    for idx, ax in enumerate(axes):
        image = arrays[idx].reshape(num_y, num_x)
        im = ax.imshow(image, **imshow_dict)
        fig.colorbar(im, ax=ax)
        if titles is not None:
            ax.set_title(titles[idx])
    
    return axes


def plot_compare_estimates(
        data_x: np.ndarray,
        data_y: np.ndarray,
        x_label: str,
        y_label: str,
        figsize: Tuple[int, int] = (5, 5)) -> None:
    _, ax = plt.subplots(figsize=figsize)

    ax.scatter(data_x, data_y)
    ax.axline((1, 1), slope=1, ls='--', c='.3')

    _max = np.max([data_x, data_y])
    ax.set_xlim(0, _max+0.1)
    ax.set_ylim(0, _max+0.1)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
