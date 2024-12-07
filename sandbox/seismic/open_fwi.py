# open-fwiを調べるために使用したコード, 結局seismic dataを見るためにはvelocity_model_gradient_calculator.pyをいじる必要があり、ここでは入力データのセットアップを行なっていただけ
# referred to https://www.devitoproject.org/examples/seismic/tutorials/01_modelling.html

from typing import List, NamedTuple, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from devito import Eq, Function, Inc, Operator, TimeFunction, norm, set_log_level, solve
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, zoom

from lib.misc import datasets_root_path
from lib.model import Vec2D
from lib.seismic import VelocityModelGradientCalculator
from lib.seismic.devito_example import AcquisitionGeometry, Receiver, SeismicModel
from lib.seismic.devito_example.acoustic import AcousticWaveSolver
from lib.visualize import show_velocity_model

# devitoのlogの抑制
set_log_level("WARNING")

WaveformData = NDArray[np.float64]
VelocityModel = NDArray[np.float64]
DampingModel = NDArray[np.float64]

WaveField = NDArray[np.float64]
WaveFieldHistory = NDArray[np.float64]


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


def psnr(signal0, signal1, max_value):
    mse = np.mean((signal0.astype(float) - signal1.astype(float)) ** 2)
    return 10 * np.log10((max_value**2) / mse)


def main():
    seismic_data_path = datasets_root_path.joinpath("open-fwi/tmp/model1.npy")
    seismic_data = np.load(seismic_data_path)

    params = Params(
        real_cell_size=Vec2D(70, 70),
        cell_meter_size=Vec2D(10.0, 10.0),
        damping_cell_thickness=40,
        start_time=0,
        unit_time=1,
        simulation_times=999,
        source_peek_time=100,
        source_frequency=0.015,
        n_shots=5,
        n_receivers=70,
    )

    shape = (params.real_cell_size.y, params.real_cell_size.x)
    spacing = (params.cell_meter_size.y, params.cell_meter_size.x)

    dsize = params.damping_cell_thickness

    target = seismic_data[0, 0] / 1000

    initial_vp = target.copy()
    for _ in range(10):
        initial_vp = gaussian_filter(initial_vp, sigma=1)

    true_model = SeismicModel(space_order=2, vp=target, origin=(0, 0), shape=shape, dtype=np.float32, spacing=spacing, nbl=params.damping_cell_thickness, bcs="damp", fs=False, dt=1)
    current_model = SeismicModel(space_order=2, vp=initial_vp, origin=(0, 0), shape=shape, dtype=np.float32, spacing=spacing, nbl=params.damping_cell_thickness, bcs="damp", fs=False, dt=1)

    print(f"initial psnr: {psnr(true_model.vp.data[dsize:-dsize, dsize:-dsize], current_model.vp.data[dsize:-dsize, dsize:-dsize], 5)}")

    # show_velocity_model(target, vmin=1.5, vmax=5)
    # show_velocity_model(true_model.vp.data[dsize:-dsize, dsize:-dsize], vmin=1.5, vmax=5)
    # show_velocity_model(current_model.vp.data[dsize:-dsize, dsize:-dsize], vmin=1.5, vmax=5)

    # src_coordinates = np.array([[30, true_model.domain_size[1] * 0.5]])
    # rec_coordinates = np.array([[30, x] for x in np.linspace(0, true_model.domain_size[1], num=params.n_receivers)])
    # source_locations = np.array([[980, x] for x in np.linspace(0.0, true_model.domain_size[1], num=params.n_shots)])

    src_coordinates = np.array([[0, 0]])
    rec_coordinates = np.array([[10, x] for x in np.linspace(0, true_model.domain_size[0], num=params.n_receivers)])
    source_locations = np.array([[10, x] for x in np.linspace(0, true_model.domain_size[0], num=params.n_shots)])

    geometry = AcquisitionGeometry(true_model, rec_coordinates, src_coordinates, params.start_time, params.simulation_times, f0=params.source_frequency, src_type="Ricker")
    solver = AcousticWaveSolver(true_model, geometry, space_order=4)

    ssolver = VelocityModelGradientCalculator(true_model, geometry, solver)

    n_iters = 1000

    def prox_l1(signal, gamma):
        return np.sign(signal) * np.maximum(np.abs(signal) - gamma, 0)

    D = lambda z: np.stack([np.roll(z, -1, axis=0) - z, np.roll(z, -1, axis=1) - z], axis=2)
    Dt = lambda z: np.roll(z[:, :, 0], 1, axis=0) - z[:, :, 0] + np.roll(z[:, :, 1], 1, axis=1) - z[:, :, 1]

    history = np.zeros((n_iters, 1))
    history1 = np.zeros((n_iters, 1))

    # flag = 'normal'
    # flag = 'admm'
    flag = "pds"

    if flag == "normal":
        step_size = 0.0026
        for i in range(0, n_iters):
            residual_norm_sum, direction = ssolver.calc_grad(current_model.vp, source_locations)

            grad = -direction / params.n_shots

            # update velocity model with box constraint
            current_model.vp.data[:] = current_model.vp.data - step_size * grad

            history[i] = residual_norm_sum
            diff = current_model.vp.data - true_model.vp.data
            history1[i] = np.sum(diff * diff)
            print(f"Objective value is {residual_norm_sum} at iteration {i + 1}, {history1[i]}")
            # if i % 100 == 0 or (i != 0 and history[i-1] < history[i]):
            # if i % 100 == 0:
            print(f"showed: {i}")
            show_velocity_model(current_model.vp.data[dsize:-dsize, dsize:-dsize], vmin=1.5, vmax=5)
            print(f"psnr: {psnr(true_model.vp.data[dsize:-dsize, dsize:-dsize], current_model.vp.data[dsize:-dsize, dsize:-dsize], 5)}")

    if flag == "admm":
        l1_norm_weight = 1
        rho = 0.001
        gamma = 0.001
        tau = 0.001
        eta = current_model.vp.data
        beta = np.zeros(eta.shape)
        for i in range(0, n_iters):
            residual_norm_sum, direction = ssolver.calc_grad(current_model.vp, source_locations)

            grad = -direction / params.n_shots

            # update velocity model with box constraint
            # step_size = 0.05 / mmax(direction)
            # current_model.vp.data[:] = np.clip(current_model.vp.data + step_size * direction.data, 2.0, 3.5)

            # current_model.vp.data[:] = np.clip(current_model.vp.data + 0.001 * direction.data, 2.0, 3.5)

            v = current_model.vp.data
            v[:] = v - rho * (grad - gamma * (eta - v - beta))
            eta = Dt(prox_l1(D(v + beta), l1_norm_weight / gamma))
            beta = beta + tau * (v - eta)

            history[i] = residual_norm_sum
            diff = current_model.vp.data - true_model.vp.data
            history1[i] = np.sum(diff * diff)
            print(f"Objective value is {residual_norm_sum} at iteration {i + 1}, {history1[i]}")
            # if i % 100 == 0 or (i != 0 and history[i-1] < history[i]):
            # if i % 100 == 0:
            print(f"showed: {i}")
            plot_velocity_model(current_model.vp.data[dsize:-dsize, dsize:-dsize].T, vmin=1.5, vmax=5)
            print(f"psnr: {psnr(true_model.vp.data[dsize:-dsize, dsize:-dsize].T, current_model.vp.data[dsize:-dsize, dsize:-dsize].T, 5)}")

    if flag == "pds":
        l1_norm_weight = 1
        gamma1 = 0.0001
        gamma2 = 100
        y = D(current_model.vp.data)
        for i in range(0, n_iters):
            residual_norm_sum, direction = ssolver.calc_grad(current_model.vp, source_locations)

            grad = -direction

            # update velocity model with box constraint
            # step_size = 0.05 / mmax(direction)
            # current_model.vp.data[:] = np.clip(current_model.vp.data + step_size * direction.data, 2.0, 3.5)

            # current_model.vp.data[:] = np.clip(current_model.vp.data + 0.001 * direction.data, 2.0, 3.5)

            prev_x = current_model.vp.data.copy()
            x = current_model.vp.data
            x[:] = x - gamma1 * (grad + Dt(y))
            y = y + gamma2 * D(2 * x - prev_x)
            y = y - gamma2 * prox_l1(y / gamma2, l1_norm_weight / gamma2)

            history[i] = residual_norm_sum
            diff = current_model.vp.data - true_model.vp.data
            history1[i] = np.sum(diff * diff)
            print(f"Objective value is {residual_norm_sum} at iteration {i + 1}, {history1[i]}")
            # if i % 100 == 0 or (i != 0 and history[i-1] < history[i]):
            # if i % 100 == 0:
            print(f"showed: {i}")
            show_velocity_model(current_model.vp.data[dsize:-dsize, dsize:-dsize], vmin=1.5, vmax=5)
            print(f"psnr: {psnr(true_model.vp.data[dsize:-dsize, dsize:-dsize], current_model.vp.data[dsize:-dsize, dsize:-dsize], 5)}")

    # Plot inverted velocity model
    # plot_velocity(current_model)

    # Plot objective function decrease
    plt.figure()
    plt.loglog(history)
    plt.xlabel("Iteration number")
    plt.ylabel("Misift value observed signal")
    plt.title("Convergence")
    plt.show()

    plt.figure()
    plt.plot(history1)
    plt.xlabel("Iteration number")
    plt.ylabel("Misift value velocity model")
    plt.title("Convergence")
    plt.show()


if __name__ == "__main__":
    main()
