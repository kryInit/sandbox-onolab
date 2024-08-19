import signal
import time
from datetime import datetime
from typing import NamedTuple, Tuple
from colorama import Fore, Style

import numpy as np
import numpy.typing as npt
from devito import set_log_level
from scipy.ndimage import gaussian_filter, zoom

from lib.dataset import load_seismic_datasets__salt_and_overthrust_models
from lib.misc import datasets_root_path, output_path
from lib.model import Vec2D
from lib.seismic import FastParallelVelocityModelGradientCalculator, FastParallelVelocityModelProps
import lib.signal_processing.diff_operator as diff_op
from lib.signal_processing.proximal_operator import proj_fast_l1_ball, prox_box_constraint
from lib.visualize import show_velocity_model

# devitoのlogの抑制
set_log_level("WARNING")


def zoom_and_crop(data: npt.NDArray, target_shape: Tuple[int, int]):
    scale = max(target_shape[0] / data.shape[0], target_shape[1] / data.shape[1])
    return zoom(data, scale)[: target_shape[0], : target_shape[1]]


def smoothing_with_gaussian_filter(data: npt.NDArray, n_iter: int, sigma: float):
    ret = data.copy()
    for _ in range(n_iter):
        ret = gaussian_filter(ret, sigma=sigma)
    return ret


def calc_psnr(signal0: npt.NDArray, signal1: npt.NDArray, max_value: float):
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

    # noise
    noise_sigma: float


def main():
    params = Params(
        real_cell_size=Vec2D(101, 51),
        cell_meter_size=Vec2D(10.0, 10.0),
        damping_cell_thickness=40,
        start_time=0,
        unit_time=1,
        simulation_times=1000,
        source_peek_time=100,
        source_frequency=0.01,
        n_shots=10,
        n_receivers=101,
        noise_sigma=5
    )

    seismic_data_path = datasets_root_path.joinpath("salt-and-overthrust-models/3-D_Salt_Model/VEL_GRIDS/Saltf@@")
    data = load_seismic_datasets__salt_and_overthrust_models(seismic_data_path).transpose((1, 0, 2)).astype(np.float32) / 1000.0

    raw_true_velocity_model = data[300]
    true_velocity_model = zoom_and_crop(raw_true_velocity_model, (51, 101))
    initial_velocity_model = smoothing_with_gaussian_filter(true_velocity_model, 20, 1)

    show_velocity_model(true_velocity_model, vmax=4.5, vmin=1.5, title="true velocity model")
    show_velocity_model(initial_velocity_model, vmax=4.5, vmin=1.5, title="initial velocity model")

    shape = (params.real_cell_size.y, params.real_cell_size.x)
    spacing = (params.cell_meter_size.y, params.cell_meter_size.x)
    dsize = params.damping_cell_thickness

    width = ((params.real_cell_size - Vec2D(1, 1)) * params.cell_meter_size).x
    source_locations = np.array([[30, x] for x in np.linspace(0, width, num=params.n_shots)])
    receiver_locations = np.array([[30, x] for x in np.linspace(0, width, num=params.n_receivers)])

    grad_calculator = FastParallelVelocityModelGradientCalculator(
        FastParallelVelocityModelProps(
            true_velocity_model,
            initial_velocity_model,
            shape,
            spacing,
            params.damping_cell_thickness,
            params.start_time,
            params.simulation_times,
            params.source_frequency,
            source_locations,
            receiver_locations,
            params.noise_sigma
        )
    )

    # algorithm = "pds"
    algorithm = "gradient"

    assert algorithm == "pds" or algorithm == "gradient"

    # l1_norm_weight = 1
    alpha = 220.07382
    gamma1 = 1e-5
    gamma2 = 0.002 if algorithm == 'pds' else None

    max_iters = 10000

    saved_state_file_name = None
    # saved_state_file_name = '2024-08-06_11-47-59,nshots=24,gamma1=1e-05,gamma2=None.npz'
    if saved_state_file_name == None:
        residual_norm_sum_history = []
        velocity_model_diff_history = []
        psnr_value_history = []

        v = grad_calculator.velocity_model.copy()
        y = diff_op.D(v)
        th = -1
    else:
        saved_state = np.load(output_path.joinpath(saved_state_file_name))
        v = saved_state['arr_0']
        y = saved_state['arr_1']
        velocity_model_diff_history = list(saved_state['arr_2'])
        residual_norm_sum_history = list(saved_state['arr_3'])
        psnr_value_history = list(saved_state['arr_4'])
        th = len(residual_norm_sum_history) - 1

    start_time = time.time()
    try:
        while True:
            th += 1
            residual_norm_sum, grad = grad_calculator.calc_grad(v)

            if algorithm == "gradient":
                v = v - gamma1 * grad

            elif algorithm == "pds":
                prev_v = v.copy()
                v = v - gamma1 * (grad + diff_op.Dt(y))
                v = prox_box_constraint(v, 1.5, 4.5)
                y = y + gamma2 * diff_op.D(2 * v - prev_v)
                y = y - gamma2 * proj_fast_l1_ball(y / gamma2, alpha)

            v_core = v[dsize:-dsize, dsize:-dsize]

            velocity_model_diff = v_core - true_velocity_model
            psnr_value = calc_psnr(true_velocity_model, v_core, 4.5)

            velocity_model_diff_history.append(np.sum(velocity_model_diff * velocity_model_diff))
            residual_norm_sum_history.append(residual_norm_sum)
            psnr_value_history.append(psnr_value)

            improved_objective = th == 0 or residual_norm_sum_history[th] < residual_norm_sum_history[th - 1]
            improved_vm_diff = th == 0 or velocity_model_diff_history[th] < velocity_model_diff_history[th - 1]
            improved_psnr = th == 0 or psnr_value_history[th] > psnr_value_history[th - 1]

            gzk = f"{Fore.GREEN}{Style.BRIGHT}↑{Fore.RESET}{Style.RESET_ALL}"
            gzj = f"{Fore.GREEN}{Style.BRIGHT}↓{Fore.RESET}{Style.RESET_ALL}"
            rzk = f"{Fore.RED}{Style.BRIGHT}↑{Fore.RESET}{Style.RESET_ALL}"
            rzj = f"{Fore.RED}{Style.BRIGHT}↓{Fore.RESET}{Style.RESET_ALL}"

            print(
                f"iters: {th+1}, objective: {residual_norm_sum_history[th]: .1f} {gzj if improved_objective else rzk}, vm diff: {velocity_model_diff_history[th]: .3f} {gzj if improved_vm_diff else rzk}, psnr: {psnr_value: .4f} {gzk if improved_psnr else rzj}"
            )
            # if (th + 1) % 100 == 0:
            #     show_velocity_model(v_core, title=f"Velocity model at iteration {th + 1}", vmax=4.5, vmin=1.5)
            if th == max_iters-1:
                break

    finally:
        # ref: https://qiita.com/qualitia_cdev/items/f536002791671c6238e3
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{current_time},nshots={params.n_shots},gamma1={gamma1},gamma2={gamma2},niters={th+1},sigma={params.noise_sigma}.npz"
        save_path = output_path.joinpath(filename)
        np.savez(save_path, v, y, np.array(velocity_model_diff_history), np.array(residual_norm_sum_history), np.array(psnr_value_history))

        v_core = v[dsize:-dsize, dsize:-dsize]
        show_velocity_model(v_core, title=f"Velocity model at iteration {th + 1}", vmax=4.5, vmin=1.5)

        print(f"elapsed: {time.time() - start_time}")
        # 子プロセスを解放
        del grad_calculator

        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)


if __name__ == "__main__":
    main()
