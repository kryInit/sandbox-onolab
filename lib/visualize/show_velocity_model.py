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
