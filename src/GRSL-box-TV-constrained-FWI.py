import signal
import time
from pathlib import Path
from typing import Literal, NamedTuple, Union

import numpy as np
import numpy.typing as npt
from devito import set_log_level
from skimage.metrics import structural_similarity as ssim

import lib.signal_processing.diff_operator as diff_op
from lib.dataset.load_BP2004_model import load_BP2004_model
from lib.misc import datasets_root_path
from lib.misc.historical_value import ValueHistoryList
from lib.model import Vec2D
from lib.seismic import FastParallelVelocityModelGradientCalculator, FastParallelVelocityModelGradientCalculatorProps
from lib.signal_processing.misc import calc_psnr, zoom_and_crop
from lib.signal_processing.norm import L12_norm
from lib.signal_processing.proximal_operator import proj_L12_norm_ball, prox_box_constraint
from lib.visualize import show_minimum_velocity_model, show_velocity_model

# devitoのlogの抑制
set_log_level("WARNING")


class FWIParams(NamedTuple):
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


class NesterovParams(NamedTuple):
    prev_v: npt.NDArray
    rho: float
    gamma: float


class VelocityModelDataForOptimization(NamedTuple):
    """
    データだけでなく、最適化に関連するパラメータも保持する(パラメータはデータに依存して変更するのでこのように持たせている, 要素が多くなるならデータとパラメータを分離したいが一旦これで
    """

    true_data: npt.NDArray
    initial_data: npt.NDArray
    box_min_value: float
    box_max_value: float


def BP2004_model_test00_configuration(n_shots: int, noise_sigma: float) -> FWIParams:
    return FWIParams(
        real_cell_size=Vec2D(276, 76),
        cell_meter_size=Vec2D(10.0, 10.0),
        damping_cell_thickness=40,
        start_time=0,
        unit_time=1,
        simulation_times=1000,
        source_peek_time=100,
        source_frequency=0.01,
        n_shots=n_shots,
        n_receivers=273,
        noise_sigma=noise_sigma,
    )


def fwi_params_to_fast_parallel_velocity_model_gradient_calculator_props(
    params: FWIParams, true_velocity_model: npt.NDArray, initial_velocity_model: npt.NDArray
) -> FastParallelVelocityModelGradientCalculatorProps:
    shape = (params.real_cell_size.y, params.real_cell_size.x)
    spacing = (params.cell_meter_size.y, params.cell_meter_size.x)
    dsize = params.damping_cell_thickness

    width = ((params.real_cell_size - Vec2D(1, 1)) * params.cell_meter_size).x
    source_locations = np.array([[30, x] for x in np.linspace(0, width, num=params.n_shots)])
    receiver_locations = np.array([[30, x] for x in np.linspace(0, width, num=params.n_receivers)])

    props = FastParallelVelocityModelGradientCalculatorProps(
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
        14,
    )
    return props


def load_BP2004_dataset(real_cell_size: Vec2D[int]):
    vmax, vmin = 4.8, 1.4

    seismic_data_path = datasets_root_path.joinpath("BP2004/vel_z6.25m_x12.5m_exact.segy")
    seismic_data = load_BP2004_model(seismic_data_path) / 1000.0
    true_velocity_model = zoom_and_crop(seismic_data[:, 1800:-1500], (real_cell_size.y * 2, real_cell_size.x))[::2]
    initial_velocity_model_col = np.linspace(vmin, vmax, num=real_cell_size.y)
    initial_velocity_model = np.tile(initial_velocity_model_col, (real_cell_size.x, 1)).T

    return VelocityModelDataForOptimization(true_velocity_model, initial_velocity_model, vmin, vmax)


def remove_damping_cells(velocity_model: npt.NDArray, damping_cell_thickness: int):
    x = damping_cell_thickness
    return velocity_model[x:-x, x:-x]


def simulate_fwi(
    max_n_iters: int,
    n_shots: int,
    noise_sigma: float,
    algorithm: Union[Literal["pds"], Literal["gradient"], Literal["gd_nesterov"], Literal["pds_nesterov"]],
    gamma1: float,
    gamma2: float,
    alpha: float,
    np_log_path: Union[Path, None] = None,
):
    np.random.seed(42)

    if algorithm == "gradient" or algorithm == "gd_nesterov":
        gamma2 = None
    if gamma2 is None and (algorithm == "pds" or algorithm == "pds_nesterov"):
        raise ValueError("gamma2 must be set when algorithm is pds")

    # connfigures
    params = BP2004_model_test00_configuration(n_shots, noise_sigma)

    # alias
    dsize = params.damping_cell_thickness

    # load data
    true_velocity_model, initial_velocity_model, vmin, vmax = load_BP2004_dataset(params.real_cell_size)

    def simple_visualize():
        show_minimum_velocity_model(np.repeat(true_velocity_model, 2, axis=0), vmax=vmax, vmin=vmin, cmap="coolwarm")
        show_minimum_velocity_model(np.repeat(initial_velocity_model, 2, axis=0), vmax=vmax, vmin=vmin, cmap="coolwarm")
        total_variation_of_true_velocity_model = L12_norm(diff_op.D(true_velocity_model))
        print(f"TV of true velocity model: {total_variation_of_true_velocity_model}, alpha: {alpha}, ratio: {alpha / total_variation_of_true_velocity_model}")

    simple_visualize()

    def create_grad_calculator():
        props = fwi_params_to_fast_parallel_velocity_model_gradient_calculator_props(params, true_velocity_model, initial_velocity_model)
        return FastParallelVelocityModelGradientCalculator(props)

    grad_calculator = create_grad_calculator()

    residual_norm_sum_values = ValueHistoryList("objective", "less", [])
    velocity_model_square_error_values = ValueHistoryList("velocity model square error", "less", [])
    psnr_values = ValueHistoryList("psnr", "greater", [])
    ssim_values = ValueHistoryList("ssim", "greater", [])
    total_variation_values = ValueHistoryList("TV", None, [])
    var_diff_values = ValueHistoryList("var diff", "less", [])

    v = grad_calculator.velocity_model.copy()
    y = diff_op.D(v)
    th = -1

    nesterov_params = NesterovParams(v.copy(), 1, 0)

    start_time = time.time()
    try:
        while True:
            th += 1
            prev_v = v.copy()

            if algorithm == "gradient":
                residual_norm_sum, grad = grad_calculator.calc_grad(v)
                if np.isnan(residual_norm_sum):
                    break
                v = v - gamma1 * grad

            elif algorithm == "pds":
                residual_norm_sum, grad = grad_calculator.calc_grad(v)
                if np.isnan(residual_norm_sum):
                    break
                v = v - gamma1 * (grad + diff_op.Dt(y))
                v = prox_box_constraint(v, vmin, vmax)
                y = y + gamma2 * diff_op.D(2 * v - prev_v)
                y = y - gamma2 * proj_L12_norm_ball(y / gamma2, alpha)

            elif algorithm == "gd_nesterov":
                prev_v = nesterov_params.prev_v
                prev_rho = nesterov_params.rho
                rho = (1 + (1 + 4 * prev_rho**2) ** 0.5) / 2
                gamma = (prev_rho - 1) / rho

                tmp_v = v + gamma * (v - prev_v)
                nesterov_params = NesterovParams(v.copy(), rho, gamma)

                residual_norm_sum, grad = grad_calculator.calc_grad(tmp_v)
                if np.isnan(residual_norm_sum):
                    break

                v = tmp_v - gamma1 * grad

            elif algorithm == "pds_nesterov":
                prev_v = nesterov_params.prev_v
                prev_rho = nesterov_params.rho
                rho = (1 + (1 + 4 * prev_rho**2) ** 0.5) / 2
                gamma = (prev_rho - 1) / rho

                tmp_v = v + gamma * (v - prev_v)
                nesterov_params = NesterovParams(v.copy(), rho, gamma)

                residual_norm_sum, grad = grad_calculator.calc_grad(tmp_v)
                if np.isnan(residual_norm_sum):
                    break

                v = tmp_v - gamma1 * (grad + diff_op.Dt(y))
                v = prox_box_constraint(v, vmin, vmax)
                y = y + gamma2 * diff_op.D(2 * v - prev_v)
                y = y - gamma2 * proj_L12_norm_ball(y / gamma2, alpha)

            v_core = remove_damping_cells(v, dsize)

            velocity_model_diff = v_core - true_velocity_model
            psnr_value = calc_psnr(true_velocity_model, v_core, vmax)
            ssim_value = ssim(true_velocity_model, v_core, data_range=vmax - vmin)
            total_variation_value = L12_norm(diff_op.D(v_core))
            var_diff = v - prev_v

            residual_norm_sum_values.append(residual_norm_sum)
            velocity_model_square_error_values.append(np.sum(velocity_model_diff * velocity_model_diff))
            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)
            total_variation_values.append(total_variation_value)
            var_diff_values.append(np.sum(var_diff * var_diff))

            print(
                f"iters: {th+1}, "
                f"{residual_norm_sum_values.prev_value_message(1)}, "
                f"{velocity_model_square_error_values.prev_value_message(3)}, "
                f"{psnr_values.prev_value_message(4)}, "
                f"{ssim_values.prev_value_message(4)}, "
                f"{total_variation_values.prev_value_message(4)}, "
                f"{var_diff_values.prev_value_message(4)}, "
            )

            if not residual_norm_sum_values.is_prev_improved():
                nesterov_params = NesterovParams(nesterov_params.prev_v, 1, 0)

            if th == max_n_iters - 1:
                break

    except Exception as e:
        print(e)
    finally:
        # ref: https://qiita.com/qualitia_cdev/items/f536002791671c6238e3
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        if np_log_path is not None:
            np.savez(
                np_log_path,
                v,
                y,
                velocity_model_square_error_values.values_as_np_array(),
                residual_norm_sum_values.values_as_np_array(),
                psnr_values.values_as_np_array(),
                ssim_values.values_as_np_array(),
                var_diff_values.values_as_np_array(),
            )

        v_core = remove_damping_cells(v, dsize)
        show_velocity_model(v_core, title=f"Velocity model at iteration {th + 1}", vmax=vmax, vmin=vmin, cmap="coolwarm")

        print(f"elapsed: {time.time() - start_time}")
        # 子プロセスを解放
        del grad_calculator

        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)


if __name__ == "__main__":
    simulate_fwi(5000, 69, 0, "pds_nesterov", 1e-4, 100, 2000)
    # ログを残したい場合は最後にパスを指定する
    # simulate_fwi(5000, 69, 0, "pds_nesterov", 1e-4, 100, 2000, Path("/home/kr/workspace/tmp.npz"))
