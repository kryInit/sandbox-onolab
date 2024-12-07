from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def show_velocity_model(data: NDArray, vmin: Union[float, None] = None, vmax: Union[float, None] = None, title: str = "velocity model", cmap: str = "jet"):
    if vmin is None:
        vmin = np.min(data)
    if vmax is None:
        vmax = np.max(data)

    plt.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("X [km]")
    plt.ylabel("Depth [km]")
    plt.show()


def show_minimum_velocity_model(data: np.ndarray, vmin: Union[float, None] = None, vmax: Union[float, None] = None, title: str = "velocity model", cmap: str = "jet"):
    if vmin is None:
        vmin = np.min(data)
    if vmax is None:
        vmax = np.max(data)

    raw_figsize = (data.shape[1], data.shape[0])
    scale = 10.0 / max(raw_figsize[0], raw_figsize[1])
    image_aspect = (raw_figsize[0] * scale, raw_figsize[1] * scale)

    fig, ax = plt.subplots(figsize=image_aspect)
    ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)

    x_max = data.shape[1] / 250
    y_max = data.shape[0] / 250

    ax.set_xticks(np.linspace(0, data.shape[1] - 1, 5))
    ax.set_xticklabels([f"{x:.1f}" for x in np.linspace(0, x_max, 5)], fontsize=22)

    ax.set_yticks(np.linspace(0, data.shape[0] - 1, 3))
    ax.set_yticklabels([f"{y:.1f}" for y in np.linspace(0, y_max, 3)], fontsize=22)

    plt.xlabel("X [km]", fontsize=22)
    plt.ylabel("Depth [km]", fontsize=22)

    plt.axis("tight")
    fig.tight_layout()
    plt.show()
