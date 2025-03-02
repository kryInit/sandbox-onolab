import signal
import time
from datetime import datetime
from typing import Literal, NamedTuple, Union

import numpy as np
import numpy.typing as npt
from colorama import Fore, Style
from devito import set_log_level
from skimage.metrics import structural_similarity as ssim

import lib.signal_processing.diff_operator as diff_op
from lib.dataset import load_seismic_datasets__salt_model
from lib.misc import datasets_root_path, output_path
from lib.model import Vec2D
from lib.seismic import FastParallelVelocityModelGradientCalculator, FastParallelVelocityModelGradientCalculatorProps
from lib.signal_processing.l_bfgs import LBFGS
from lib.signal_processing.misc import calc_psnr, smoothing_with_gaussian_filter, zoom_and_crop
from lib.signal_processing.norm import L12_norm
from lib.signal_processing.proximal_operator import proj_fast_l1_ball, proj_L12_norm_ball, prox_box_constraint
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


class NesterovParams(NamedTuple):
    prev_v: npt.NDArray
    rho: float
    gamma: float


def simulate_fwi(
    max_n_iters: int,
    n_shots: int,
    noise_sigma: float,
    algorithm: Union[Literal["pds"], Literal["gradient"], Literal["gd_nesterov"], Literal["pds_nesterov"], Literal["pds_with_L12norm"], Literal["L-BFGS"]],
    gamma1: float,
    gamma2: float,
    parent_file_path: str | None,
    alpha: float,
):
    if algorithm == "gradient" or algorithm == "gd_nesterov" or algorithm == "L-BFGS":
        gamma2 = None

    params = Params(
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

    vmax, vmin = 4.5, 1.5
    # vmax, vmin = 5.8, 2

    seismic_data_path = datasets_root_path.joinpath("salt-and-overthrust-models/3-D_Salt_Model/VEL_GRIDS/Saltf@@")
    seismic_data = load_seismic_datasets__salt_model(seismic_data_path).transpose((1, 0, 2)).astype(np.float32) / 1000.0
    assert 1.5 <= np.min(seismic_data) and np.max(seismic_data) <= 4.5

    raw_true_velocity_model = seismic_data[300]
    true_velocity_model = zoom_and_crop(raw_true_velocity_model, (params.real_cell_size.y, params.real_cell_size.x))
    initial_velocity_model = zoom_and_crop(smoothing_with_gaussian_filter(seismic_data[300], 1, 80), (params.real_cell_size.y, params.real_cell_size.x))

    # seismic_data_path = datasets_root_path.joinpath("salt-and-overthrust-models/3-D_Overthrust_Model_Disk1/3D-Velocity-Grid/overthrust.vites")
    # nx, ny, nz = 801, 801, 187
    # with open(seismic_data_path, "r") as file:
    #     vel = np.fromfile(file, dtype=np.dtype("float32").newbyteorder(">"))
    #     vel = vel.reshape(nx, ny, nz, order="F")
    #     vel = np.asarray(vel, dtype=float)
    #     vel = np.flip(vel, 2)
    #     seismic_data = np.transpose(vel, (2, 1, 0))
    #     seismic_data[:] = seismic_data[::-1]
    #
    # true_velocity_model = zoom_and_crop(seismic_data[:-50, 444, 100:], (params.real_cell_size.y, params.real_cell_size.x)) / 1000
    # initial_velocity_model = zoom_and_crop(smoothing_with_gaussian_filter(seismic_data[:, 444], 1, 80)[:-50, 100:], (params.real_cell_size.y, params.real_cell_size.x)) / 1000

    # show_minimum_velocity_model(true_velocity_model, vmax=vmax, vmin=vmin, title="true velocity model", cmap='coolwarm')
    # show_minimum_velocity_model(initial_velocity_model, vmax=vmax, vmin=vmin, title="initial velocity model", cmap='coolwarm')

    # import sys
    # sys.exit(-1)

    shape = (params.real_cell_size.y, params.real_cell_size.x)
    spacing = (params.cell_meter_size.y, params.cell_meter_size.x)
    dsize = params.damping_cell_thickness

    width = ((params.real_cell_size - Vec2D(1, 1)) * params.cell_meter_size).x
    source_locations = np.array([[30, x] for x in np.linspace(0, width, num=params.n_shots)])
    receiver_locations = np.array([[30, x] for x in np.linspace(0, width, num=params.n_receivers)])

    grad_calculator = FastParallelVelocityModelGradientCalculator(
        FastParallelVelocityModelGradientCalculatorProps(
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
    )

    # l1_norm_weight = 1
    # alpha = (np.sum(np.abs(diff_op.D(true_velocity_model))) if algorithm != "pds_with_L12norm" else L12_norm(diff_op.D(true_velocity_model))) / 2
    print(np.sum(np.abs(diff_op.D(true_velocity_model))), L12_norm(diff_op.D(true_velocity_model)), alpha)

    if parent_file_path is None:
        residual_norm_sum_history = []
        velocity_model_diff_history = []
        psnr_value_history = []
        ssim_value_history = []
        var_diff_history = []

        v = grad_calculator.velocity_model.copy()
        y = diff_op.D(v)
        th = -1
    else:
        saved_state = np.load(output_path.joinpath(parent_file_path))
        v = saved_state["arr_0"]
        y = saved_state["arr_1"]
        velocity_model_diff_history = list(saved_state["arr_2"])
        residual_norm_sum_history = list(saved_state["arr_3"])
        psnr_value_history = list(saved_state["arr_4"])
        ssim_value_history = list(saved_state["arr_5"])
        var_diff_history = list(saved_state["arr_6"])
        th = len(residual_norm_sum_history) - 1

    nesterov_params = NesterovParams(v.copy(), 1, 0)
    lbfgs = LBFGS(step_size=gamma1, m=2)

    start_time = time.time()
    try:
        while True:
            th += 1
            prev_v = v.copy()
            # tmp = np.sum(grad * grad)
            # gamma1 = np.sqrt(((500 * (0.98 ** th)) + 20 * (0.9995 ** th)) / tmp)


            if algorithm == "gradient":
                residual_norm_sum, grad = grad_calculator.calc_grad(v)
                if np.isnan(residual_norm_sum):
                    break
                v = v - gamma1 * grad

            elif algorithm == "pds_with_L12norm":
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

            v_core = v[dsize:-dsize, dsize:-dsize]

            velocity_model_diff = v_core - true_velocity_model
            psnr_value = calc_psnr(true_velocity_model, v_core, vmax)
            ssim_value = ssim(true_velocity_model, v_core, data_range=vmax - vmin)
            var_diff = v - prev_v

            velocity_model_diff_history.append(np.sum(velocity_model_diff * velocity_model_diff))
            residual_norm_sum_history.append(residual_norm_sum)
            psnr_value_history.append(psnr_value)
            ssim_value_history.append(ssim_value)
            var_diff_history.append(np.sum(var_diff * var_diff))

            improved_objective = th == 0 or residual_norm_sum_history[th] < residual_norm_sum_history[th - 1]
            improved_vm_diff = th == 0 or velocity_model_diff_history[th] < velocity_model_diff_history[th - 1]
            improved_psnr = th == 0 or psnr_value_history[th] > psnr_value_history[th - 1]
            improved_ssim = th == 0 or ssim_value_history[th] > ssim_value_history[th - 1]
            decrease_var_diff = th == 0 or var_diff_history[th] < var_diff_history[th - 1]

            gzk = f"{Fore.GREEN}{Style.BRIGHT}↑{Fore.RESET}{Style.RESET_ALL}"
            gzj = f"{Fore.GREEN}{Style.BRIGHT}↓{Fore.RESET}{Style.RESET_ALL}"
            rzk = f"{Fore.RED}{Style.BRIGHT}↑{Fore.RESET}{Style.RESET_ALL}"
            rzj = f"{Fore.RED}{Style.BRIGHT}↓{Fore.RESET}{Style.RESET_ALL}"

            tv = L12_norm(diff_op.D(v))

            if not improved_objective:
                nesterov_params = NesterovParams(nesterov_params.prev_v, 1, 0)

            print(
                f"iters: {th+1}, "
                f"objective: {residual_norm_sum_history[th]: .1f} {gzj if improved_objective else rzk}, "
                f"vm diff: {velocity_model_diff_history[th]: .3f} {gzj if improved_vm_diff else rzk}, "
                f"psnr: {psnr_value: .4f} {gzk if improved_psnr else rzj}, "
                f"ssim: {ssim_value: .4f} {gzk if improved_ssim else rzj}, "
                f"TV: {tv: .4f}, "
                f"var diff: {var_diff_history[th]: .4f} {gzj if decrease_var_diff else rzj}, "
                f"rho: {nesterov_params.rho: .4f}, "
                f"gamma: {nesterov_params.gamma: .4f}, "
            )
            # show_velocity_model(grad[dsize:-dsize, dsize:-dsize], title=f"Velocity model at iteration {th + 1}", cmap='coolwarm')
            # show_velocity_model(v_core, title=f"Velocity model at iteration {th + 1}", vmax=vmax, vmin=vmin, cmap='coolwarm')
            # if (th + 1) % 100 == 0:
            #     show_velocity_model(v, title=f"Velocity model at iteration {th + 1}", vmax=vmax, vmin=vmin, cmap='coolwarm')
            # show_velocity_model(v_core, title=f"Velocity model at iteration {th + 1}", vmax=vmax, vmin=vmin, cmap='coolwarm')
            if th == max_n_iters - 1:
                break
    except Exception as e:
        print(e)
    finally:
        # ref: https://qiita.com/qualitia_cdev/items/f536002791671c6238e3
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{current_time},{algorithm},nshots={params.n_shots},gamma1={gamma1},gamma2={gamma2},niters={th+1},sigma={params.noise_sigma},alpha={alpha},.npz"
        save_path = output_path.joinpath(filename)
        # np.savez(save_path, v, y, np.array(velocity_model_diff_history), np.array(residual_norm_sum_history), np.array(psnr_value_history), np.array(ssim_value_history), np.array(var_diff_history))

        # v_core = v[dsize:-dsize, dsize:-dsize]
        # show_velocity_model(v_core, title=f"Velocity model at iteration {th + 1}", vmax=vmax, vmin=vmin, cmap="coolwarm")

        print(f"elapsed: {time.time() - start_time}")
        # 子プロセスを解放
        del grad_calculator

        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)


if __name__ == "__main__":
    np.random.seed(42)

    simulate_fwi(2000, 69, 0, "pds_nesterov", 1e-5, 100, None, 1200)
    # simulate_fwi(2000, 69, 10, "pds_nesterov", 1e-5, 100, None, 1200)
    # simulate_fwi(2000, 69, 1, "gd_nesterov", 1e-5, 100, None, 1200)
    # simulate_fwi(2000, 69, 5, "gd_nesterov", 1e-5, 100, None, 1200)
    # simulate_fwi(2000, 69, 3, "gd_nesterov", 1e-5, 100, None, 1200)
    # simulate_fwi(2000, 69, 10, "gd_nesterov", 1e-5, 100, None, 1200)
    # simulate_fwi(2000, 69, 0, "pds_with_L12norm", 1e-5, 100, None, 1200)
    # simulate_fwi(2000, 69, 0, "gradient", 1e-5, 100, None, 0)

