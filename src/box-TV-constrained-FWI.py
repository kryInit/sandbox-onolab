import signal
import time
from datetime import datetime
from typing import NamedTuple, Literal, Union

import numpy as np
import numpy.typing as npt
from devito import set_log_level
from skimage.metrics import structural_similarity as ssim

from lib.dataset import load_seismic_datasets__salt_model
from lib.dataset.load_overthrust_model import load_seismic_datasets__overthrust_model
from lib.misc import datasets_root_path, output_path
from lib.misc.historical_value import HistoricalValue
from lib.model import Vec2D
from lib.seismic import FastParallelVelocityModelGradientCalculator, FastParallelVelocityModelProps
from lib.signal_processing.misc import zoom_and_crop, smoothing_with_gaussian_filter, calc_psnr
from lib.signal_processing.norm import L12_norm
import lib.signal_processing.diff_operator as diff_op
from lib.signal_processing.proximal_operator import prox_box_constraint, proj_L12_norm_ball
from lib.visualize import show_velocity_model

# devitoのlogの抑制
set_log_level("WARNING")


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


class VelocityModelDataForOptimization(NamedTuple):
    """
    データだけでなく、最適化に関連するパラメータも保持する(パラメータはデータに依存して変更するのでこのように持たせている, 要素が多くなるならデータとパラメータを分離したいが一旦これで
    """
    true_data: npt.NDArray
    initial_data: npt.NDArray
    box_min_value: float
    box_max_value: float


def load_salt_model(real_cell_size: Vec2D[int], target_idx: int = 300):
    vmin, vmax = 1.5, 4.5

    seismic_data_path = datasets_root_path.joinpath("salt-and-overthrust-models/3-D_Salt_Model/VEL_GRIDS/Saltf@@")
    seismic_data = load_seismic_datasets__salt_model(seismic_data_path).transpose((1, 0, 2)).astype(np.float32) / 1000.0
    assert vmin <= np.min(seismic_data) and np.max(seismic_data) <= vmax

    raw_true_velocity_model = seismic_data[target_idx]
    true_velocity_model = zoom_and_crop(raw_true_velocity_model, (real_cell_size.y, real_cell_size.x))
    initial_velocity_model = zoom_and_crop(smoothing_with_gaussian_filter(seismic_data[target_idx], 1, 80), (real_cell_size.y, real_cell_size.x))

    return VelocityModelDataForOptimization(true_velocity_model, initial_velocity_model, vmin, vmax)


def load_overthrust_model(real_cell_size: Vec2D[int], target_idx: int = 300):
    vmin, vmax = 2.1, 6.0

    seismic_data_path = datasets_root_path.joinpath("salt-and-overthrust-models/3-D_Overthrust_Model_Disk1/3D-Velocity-Grid/overthrust.vites")
    seismic_data = load_seismic_datasets__overthrust_model(seismic_data_path).transpose((1, 0, 2)).astype(np.float32) / 1000.0
    assert vmin <= np.min(seismic_data) and np.max(seismic_data) <= vmax

    raw_true_velocity_model = seismic_data[target_idx]
    true_velocity_model = zoom_and_crop(raw_true_velocity_model, (real_cell_size.y, real_cell_size.x))
    initial_velocity_model = zoom_and_crop(smoothing_with_gaussian_filter(seismic_data[target_idx], 1, 80), (real_cell_size.y, real_cell_size.x))

    return VelocityModelDataForOptimization(true_velocity_model, initial_velocity_model, vmin, vmax)


def remove_damping_cells(velocity_model: npt.NDArray, damping_cell_thickness: int):
    x = damping_cell_thickness
    return velocity_model[x:-x, x:-x]


def simulate_fwi(max_n_iters: int, n_shots: int, noise_sigma: float, algorithm: Union[Literal['pds'], Literal['gradient']], gamma1: float, gamma2: float, alpha: float):
    if algorithm == "gradient":
        gamma2 = None

    # configures
    params = Params(
        real_cell_size=Vec2D(100, 50),
        cell_meter_size=Vec2D(10.0, 10.0),
        damping_cell_thickness=40,
        start_time=0,
        unit_time=1,
        simulation_times=1000,
        source_peek_time=100,
        source_frequency=0.01,
        n_shots=n_shots,
        n_receivers=101,
        noise_sigma=noise_sigma
    )

    # alias
    dsize = params.damping_cell_thickness

    # load data
    true_velocity_model, initial_velocity_model, vmin, vmax = load_salt_model(params.real_cell_size)

    # simple visualize
    def simple_visualize():
        show_velocity_model(true_velocity_model, vmax=vmax, vmin=vmin, title="true velocity model", cmap='coolwarm')
        show_velocity_model(initial_velocity_model, vmax=vmax, vmin=vmin, title="initial velocity model", cmap='coolwarm')
        total_variation_of_true_velocity_model = L12_norm(diff_op.D(true_velocity_model))
        print(f"TV of true velocity model: {total_variation_of_true_velocity_model}, alpha: {alpha}, ratio: {alpha / total_variation_of_true_velocity_model}")
    simple_visualize()

    def create_grad_calculator():
        shape = (params.real_cell_size.y, params.real_cell_size.x)
        spacing = (params.cell_meter_size.y, params.cell_meter_size.x)
        width = ((params.real_cell_size - Vec2D(1, 1)) * params.cell_meter_size).x
        source_locations = np.array([[30, x] for x in np.linspace(0, width, num=params.n_shots)])
        receiver_locations = np.array([[30, x] for x in np.linspace(0, width, num=params.n_receivers)])
        return FastParallelVelocityModelGradientCalculator(
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
                params.noise_sigma,
                20
            )
        )
    grad_calculator = create_grad_calculator()

    residual_norm_sum_values = HistoricalValue("objective", "less", [])
    velocity_model_square_error_values = HistoricalValue("velocity model square error", "less", [])
    psnr_values = HistoricalValue("psnr", "greater", [])
    ssim_values = HistoricalValue("ssim", "greater", [])
    total_variation_values = HistoricalValue("TV", None, [])

    v = grad_calculator.velocity_model.copy()
    y = diff_op.D(remove_damping_cells(v, dsize))
    th = -1

    start_time = time.time()
    try:
        while True:
            th += 1
            residual_norm_sum, grad = grad_calculator.calc_grad(v)

            if np.isnan(residual_norm_sum):
                break

            if algorithm == "gradient":
                v = v - gamma1 * grad

            elif algorithm == "pds":
                prev_v = v.copy()

                tmp = grad.copy()
                tmp[dsize:-dsize, dsize:-dsize] += diff_op.Dt(y)
                v = v - gamma1 * tmp
                v[dsize:-dsize, dsize:-dsize] = prox_box_constraint(remove_damping_cells(v, dsize), vmin, vmax)
                y = y + gamma2 * diff_op.D(2 * remove_damping_cells(v, dsize) - remove_damping_cells(prev_v, dsize))
                y = y - gamma2 * proj_L12_norm_ball(y / gamma2, alpha)

            v_core = remove_damping_cells(v, dsize)

            velocity_model_diff = v_core - true_velocity_model
            velocity_model_square_error = np.sum(velocity_model_diff * velocity_model_diff)
            psnr_value = calc_psnr(true_velocity_model, v_core, vmax)
            ssim_value = ssim(true_velocity_model, v_core, data_range=vmax - vmin)
            total_variation_value = L12_norm(diff_op.D(v_core))

            residual_norm_sum_values.append(residual_norm_sum)
            velocity_model_square_error_values.append(velocity_model_square_error)
            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)
            total_variation_values.append(total_variation_value)

            print(
                f"iters: {th+1}, "
                f"{residual_norm_sum_values.prev_value_message(1)}, "
                f"{velocity_model_square_error_values.prev_value_message(3)}, "
                f"{psnr_values.prev_value_message(4)}, "
                f"{ssim_values.prev_value_message(4)}, "
                f"{total_variation_values.prev_value_message(4)}, "
            )
            # show_velocity_model(grad[dsize:-dsize, dsize:-dsize], title=f"Velocity model at iteration {th + 1}", cmap='coolwarm')
            # show_velocity_model(v_core, title=f"Velocity model at iteration {th + 1}", vmax=vmax, vmin=vmin, cmap='coolwarm')
            # if (th + 1) % 1000 == 0:
                # show_velocity_model(v, title=f"Velocity model at iteration {th + 1}", vmax=vmax, vmin=vmin, cmap='coolwarm')
                # show_velocity_model(v_core, title=f"Velocity model at iteration {th + 1}", vmax=vmax, vmin=vmin, cmap='coolwarm')
            if th == max_n_iters-1:
                break

    finally:
        # ref: https://qiita.com/qualitia_cdev/items/f536002791671c6238e3
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{current_time},{algorithm},nshots={params.n_shots},gamma1={gamma1},gamma2={gamma2},niters={th+1},sigma={params.noise_sigma},alpha={alpha},.npz"
        save_path = output_path.joinpath(filename)
        # np.savez(save_path, v, y, np.array(velocity_model_diff_history), np.array(residual_norm_sum_history), np.array(psnr_value_history), np.array(ssim_value_history))

        v_core = remove_damping_cells(v, dsize)
        show_velocity_model(v_core, title=f"Velocity model at iteration {th + 1}", vmax=vmax, vmin=vmin, cmap='coolwarm')

        print(f"elapsed: {time.time() - start_time}")
        # 子プロセスを解放
        del grad_calculator

        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)


if __name__ == "__main__":
    # simulate_fwi(5000, 20, 0, "pds", 1e-4, 100, None, 2000)
    # simulate_fwi(5000, 20, 1, "gradient", 1e-4, 100, None, 550)
    simulate_fwi(100, 20, 1, "pds", 1e-4, 100, 350)
    # simulate_fwi(5000, 20, 1, "pds", 1e-4, 100, None, 150)
    # simulate_fwi(5000, 20, 1, "pds", 1e-4, 100, None, 200)
    # simulate_fwi(5000, 20, 1, "pds", 1e-4, 100, None, 250)
    # simulate_fwi(5000, 20, 1, "pds", 1e-4, 100, None, 300)
    # simulate_fwi(5000, 20, 1, "pds", 1e-4, 100, None, 350)
    # simulate_fwi(5000, 20, 1, "pds", 1e-4, 100, None, 400)
    # simulate_fwi(5000, 20, 1, "pds", 1e-4, 100, None, 450)
    # simulate_fwi(5000, 20, 1, "pds", 1e-4, 100, None, 500)
    # simulate_fwi(5000, 20, 1, "pds", 1e-4, 100, None, 550)
    # simulate_fwi(5000, 20, 1, "pds", 1e-4, 100, None, 600)
    # simulate_fwi(5000, 20, 1, "pds", 1e-4, 100, None, 650)
    # simulate_fwi(5000, 20, 1, "pds", 1e-4, 100, None, 700)
    # simulate_fwi(5000, 20, 0, "gradient", 1e-4, 100, None, 0)
    # simulate_fwi(2000, 20, 0, "gradient", 1e-4, 100, None)



    # simulate_fwi(60000, 24, 1, "pds", 1e-5, 0.01, '2024-08-09_01-50-11,nshots=24,gamma1=1e-05,gamma2=0.01,niters=30000,sigma=1.npz')

