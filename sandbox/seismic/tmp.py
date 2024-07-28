import datetime
import os
import time
from typing import NamedTuple, Tuple
import numpy as np
import numpy.typing as npt
from devito import set_log_level
from scipy.ndimage import gaussian_filter, zoom

from lib.dataset import load_seismic_datasets__salt_and_overthrust_models
from lib.misc import datasets_root_path, output_path
from lib.model import Vec2D
from lib.seismic.devito_example import AcquisitionGeometry, SeismicModel
from lib.seismic.devito_example.acoustic import AcousticWaveSolver
from lib.seismic.fast_parallel_velocity_model_gradient_calculator import FastParallelVelocityModelGradientCalculator, FastParallelVelocityModelProps
from lib.seismic.velocity_model_gradient_calculator import VelocityModelGradientCalculator
from lib.visualize.show_velocity_model import show_velocity_model

# devitoのlogの抑制
set_log_level("WARNING")


def D(x: npt.NDArray):
    diff1 = np.concatenate((x[1:], x[-1:]), axis=0) - x
    diff2 = np.concatenate((x[:, 1:], x[:, -1:]), axis=1) - x
    result = np.stack((diff1, diff2), axis=2)
    return result


def Dt(y: npt.NDArray):
    ret0 = np.concatenate([-y[0:1, :, 0], -y[1:-1, :, 0] + y[:-2, :, 0], y[-2:-1, :, 0]], axis=0)
    ret1 = np.concatenate([-y[:, 0:1, 1], -y[:, 1:-1, 1] + y[:, :-2, 1], y[:, -2:-1, 1]], axis=1)
    return ret0 + ret1


def L12_norm(signal: npt.NDArray):
    return np.sum(np.sqrt(np.sum(signal**2, axis=2)))


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


def zoom_and_crop(data: npt.NDArray, target_shape: Tuple[int, int]):
    scale = max(target_shape[0] / data.shape[0], target_shape[1] / data.shape[1])
    return zoom(data, scale)[: target_shape[0], : target_shape[1]]


def smoothing_with_gaussian_filter(data: npt.NDArray, n_iter: int, sigma: float):
    ret = data.copy()
    for _ in range(n_iter):
        ret = gaussian_filter(ret, sigma=sigma)
    return ret


def psnr(signal0: npt.NDArray, signal1: npt.NDArray, max_value: float):
    mse = np.mean((signal0.astype(float) - signal1.astype(float)) ** 2)
    return 10 * np.log10((max_value**2) / mse)


class Params(NamedTuple):
    real_cell_size: Vec2D[int]
    cell_meter_size: Vec2D[float]

    # damping
    damping_cell_thickness: int

    # time
    start_time: float
    unit_time: float
    simulation_times: int

    # input source
    source_peek_time: float
    source_frequency: float

    # shorts
    n_shots: int
    n_receivers: int


print("HI")

params = Params(
    real_cell_size=Vec2D(101, 51),
    cell_meter_size=Vec2D(10.0, 10.0),
    damping_cell_thickness=40,
    start_time=0,
    unit_time=1,
    simulation_times=1000,
    source_peek_time=100,
    source_frequency=0.01,
    n_shots=26,
    n_receivers=101,
)

seismic_data_path = datasets_root_path.joinpath("salt-and-overthrust-models/3-D_Salt_Model/VEL_GRIDS/Saltf@@")
data = load_seismic_datasets__salt_and_overthrust_models(seismic_data_path).transpose((1, 0, 2)).astype(np.float32) / 1500.0

raw_true_velocity_model = data[300]
true_velocity_model = zoom_and_crop(raw_true_velocity_model, (51, 101))
initial_velocity_model = smoothing_with_gaussian_filter(true_velocity_model, 10, 1)

# show_velocity_model(raw_true_velocity_model, title="Raw true velocity model", vmax=3, vmin=1)
# show_velocity_model(true_velocity_model, title="True velocity model", vmax=3, vmin=1)
# show_velocity_model(initial_velocity_model, title="Initial velocity model", vmax=3, vmin=1)

shape = (params.real_cell_size.x, params.real_cell_size.y)
spacing = (params.cell_meter_size.x, params.cell_meter_size.y)

width = ((params.real_cell_size - Vec2D(1, 1)) * params.cell_meter_size).x
source_locations = np.array([[x, 30] for x in np.linspace(0, width, num=params.n_shots)])
receiver_locations = np.array([[x, 30] for x in np.linspace(0, width, num=params.n_receivers)])


# true_model = SeismicModel(space_order=2, vp=true_velocity_model.T, origin=(0, 0), shape=shape, dtype=np.float32, spacing=spacing, nbl=params.damping_cell_thickness, bcs="damp", fs=False)
current_model = SeismicModel(space_order=2, vp=initial_velocity_model.T, origin=(0, 0), shape=shape, dtype=np.float32, spacing=spacing, nbl=params.damping_cell_thickness, bcs="damp", fs=False)
# geometry = AcquisitionGeometry(true_model, receiver_locations, np.array([[0, 0]]), params.start_time, params.simulation_times, f0=params.source_frequency, src_type="Ricker")
# simulator = AcousticWaveSolver(true_model, geometry, space_order=4)
# grad_calculator = VelocityModelGradientCalculator(
#     true_model,
#     geometry,
#     simulator
# )

grad_calculator = FastParallelVelocityModelGradientCalculator(FastParallelVelocityModelProps(
    true_velocity_model,
    initial_velocity_model,
    shape,
    spacing,
    params.damping_cell_thickness,
    params.start_time,
    params.simulation_times,
    params.source_frequency,
    source_locations,
    receiver_locations
))
print("initialized")

# l1_norm_weight = 1
alpha = 220.07382
gamma1 = 0.00001
gamma2 = 0.0001

residual_norm_sum_history = np.zeros(0)
velocity_model_diff_history = np.zeros(0)

dsize = params.damping_cell_thickness
vm_shape = (params.real_cell_size[0] + dsize * 2, params.real_cell_size[1] + dsize * 2)
v = current_model.vp.data.copy()
# v = current_model.vp.data
y = D(v)
th = -1
# try:

start_time = time.time()
while True:
    th += 1
    # residual_norm_sum, grad = grad_calculator.calc_grad(current_model.vp, source_locations)
    dsize = params.damping_cell_thickness
    residual_norm_sum, grad = grad_calculator.calc_grad(v)

    prev_v = v.copy()

    v[:] = v - gamma1 * grad
    # v[:] = v - gamma1 * (grad + Dt(y))
    # v[:] = prox_box_constraint(v, 1, 3)
    # y = y + gamma2 * D(2 * v - prev_v)
    # y = y - gamma2 * proj_fast_l1_ball(y / gamma2, alpha)

    v_core = v[dsize:-dsize, dsize:-dsize].T

    velocity_model_diff = v_core - true_velocity_model
    velocity_model_diff_history = np.append(velocity_model_diff_history, np.sum(velocity_model_diff * velocity_model_diff))
    residual_norm_sum_history = np.append(residual_norm_sum_history, residual_norm_sum)

    improved_objective = th == 0 or residual_norm_sum_history[th] < residual_norm_sum_history[th - 1]
    improved_vm_diff = th == 0 or velocity_model_diff_history[th] < velocity_model_diff_history[th - 1]
    print(
        f"iters: {th+1}, objective: {residual_norm_sum_history[th]: .1f} {"↓" if improved_objective else "↑"}, vm diff: {velocity_model_diff_history[th]: .3f} {"↓" if improved_vm_diff else "↑"}, psnr: {psnr(true_velocity_model, v_core, 3): .4f}"
    )
    # if th % 10 == 0:
        # show_velocity_model(v_core, title=f"Velocity model at iteration {th + 1}", vmax=3, vmin=1)
    if th == 100:
        break
# except:
    # current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # filename = f"gamma1={gamma1},gamma2={gamma2},{current_time}.npz"
    # save_path = output_path.joinpath(filename)
    # np.savez(save_path, v, y, velocity_model_diff_history, residual_norm_sum_history)

v_core = v[dsize:-dsize, dsize:-dsize].T
show_velocity_model(v_core, title=f"Velocity model at iteration {10000}", vmax=3, vmin=1)

print(f"elapsed: {time.time() - start_time}")
# 子プロセスを解放
del grad_calculator
