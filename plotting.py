from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
import numpy as np

# CAR Plotting


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

    return fig, axes, imshow_dict


def plot_compare_estimates(
        data_x: np.ndarray,
        data_y: np.ndarray,
        x_label: str,
        y_label: str,
        figsize: Tuple[int, int] = (5, 5),
        vmin: Optional[int] = None,
        vmax: Optional[int] = None,
        xlim: Optional[Tuple[int, int]] = None,
        ylim: Optional[Tuple[int, int]] = None) -> None:
    _, ax = plt.subplots(figsize=figsize)

    ax.scatter(data_x, data_y)
    ax.axline((1, 1), slope=1, ls='--', c='.3')

    if vmax is None:
        vmax = np.max([data_x, data_y])

    if xlim is None:
        xlim = (0, vmax+0.1)
    if ylim is None:
        ylim = (0, vmax+0.1)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

# 1DGP Plotting


def plot_gp_draws(x: np.ndarray, draws: np.ndarray,
                  num_draws_to_plot=None,
                  title=None, x_label=None, y_label=None, ax=None) -> None:
    if x_label is None:
        x_label = '$x$'
    if num_draws_to_plot is None:
        num_draws_to_plot = len(draws)

    if ax is None:
        _, ax = plt.subplots()

    for batch in draws[:num_draws_to_plot]:
        ax.plot(x, np.squeeze(batch))

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # NOTE :- Do not end with `plot.show()` since we allow multiple axes.


def plot_gp_draws_with_stats(
        x: np.ndarray, draws: np.ndarray,
        mean_draw: np.ndarray, hpdi_draw: np.ndarray,
        alpha: float, num_draws_to_plot: int,
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        y_lim: Tuple[float, float] = None,
        ax: Optional[plt.Axes] = None,) -> None:

    if y_lim is None:
        y_lim = [-2, 2]
    if x_label is None:
        x_label = '$x$'

    if ax is None:
        _, ax = plt.subplots()

    for j in range(num_draws_to_plot):
        ax.plot(x, draws[j], alpha=alpha, color="darkgreen", label="")
    ax.plot(x, draws[0], alpha=alpha, color="darkgreen", label="draws")
    ax.fill_between(x, hpdi_draw[0], hpdi_draw[1],
                    alpha=0.1, interpolate=True, label="95% HPDI")
    ax.plot(x, mean_draw, label="mean")
    ax.set_ylim(y_lim)

    ax.set_xlabel('$x$')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc=4)


def convert_to_string_int(value_int):
    # Helper function for clean print of hyperparams.
    if value_int >= 1_000_000:
        value_string = f'{value_int / 1_000_000}M'
    elif value_int >= 1000:
        value_string = f'{value_int // 1000}K'
    else:
        value_string = f'{value_int}'
    return value_string
