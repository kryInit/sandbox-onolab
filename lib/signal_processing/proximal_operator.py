import numpy as np
from numpy import typing as npt


def prox_12_band(signal, gamma):
    norm = np.sqrt(np.sum(signal**2, axis=2)) + 1e-8
    tmp = np.maximum(1 - gamma / norm, 0)
    return tmp[..., np.newaxis] * signal


def proj_fast_l1_ball(signal, alpha):
    x = signal.flatten()
    abs_x = np.abs(x)
    tmp = np.maximum((np.cumsum(np.sort(abs_x)[::-1]) - alpha) / np.arange(1, len(x) + 1), 0)
    x = np.maximum(abs_x - tmp.max(), 0) * np.sign(x)
    return np.reshape(x, signal.shape)


def prox_box_constraint(signal: npt.NDArray[float], l: float, r: float) -> npt.NDArray:
    return np.clip(signal, l, r)
