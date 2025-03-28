from pathlib import Path

import numpy as np
from numpy import typing as npt
from obspy.io.segy.segy import _read_segy

from lib.misc.paths import datasets_root_path


def load_BP2004_model(data_path: Path) -> npt.NDArray[np.float64]:
    segy_data = _read_segy(data_path)
    traces = [trace.data for trace in segy_data.traces]
    velocity_model = np.array(traces).T

    nz, nx = 1911, 5395
    assert velocity_model.shape == (nz, nx)
    assert np.min(velocity_model) == 1429.000244140625
    assert np.max(velocity_model) == 4790.0

    return velocity_model
